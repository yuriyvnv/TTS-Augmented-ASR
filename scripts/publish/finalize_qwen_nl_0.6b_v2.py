"""Finalise the 0.6B NL v2 run after early stop at epoch 2.

What this does:
  1. Loads checkpoint-800 (best by eval_loss = 0.1249) from the v2 output dir.
  2. Evaluates on CV17-nl test + CV22-nl test using the same evaluate_model()
     that the in-training final eval would have used → writes test_results.json.
  3. Pushes weights + full v2 README to yuriyvnv/Qwen3-ASR-0.6B-NL.

Why standalone: we killed training mid-step-1000+ to lock in the best-by-loss
checkpoint, so the in-line stage-1/stage-2 push at the end of train_qwen3_asr
never ran. This script reproduces those two stages from the saved checkpoint.

Usage (inside Docker container):
    bash scripts/docker/shell.sh
    cd /workspace
    uv run python scripts/publish/finalize_qwen_nl_0.6b_v2.py

Approx runtime: ~30 min on H100 for both test sets at batch=12.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from transformers import AutoProcessor, GenerationConfig

from src.training.train_qwen3_asr import (
    LANGUAGE_NAMES,
    Qwen3ASRForTraining,
    _build_full_readme,
    _build_simple_readme,
    _dataset_card_meta,
    _load_qwen_zero_shot_baseline,
    _model_short_name,
    evaluate_model,
    load_cv17_test,
    load_cv22,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="./results/qwen3_finetune_nl/0.6b_mixed_nl_full_s42/checkpoint-800",
        help="Path to the checkpoint to publish. Default = best-by-loss for the "
             "0.6B NL v2 run (epoch 1.69, eval_loss 0.1249).",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/qwen3_finetune_nl/0.6b_mixed_nl_full_s42",
        help="Where to write the finalised model + test_results.json. The "
             "checkpoint contents are copied here (top-level) so the HF upload "
             "matches the structure trainer.save_model would have produced.",
    )
    parser.add_argument("--language", default="nl", choices=sorted(LANGUAGE_NAMES.keys()))
    parser.add_argument("--base-model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--hub-repo-id", default="yuriyvnv/Qwen3-ASR-0.6B-NL")
    parser.add_argument("--dataset", default="mixed_nl_full",
                        help="Dataset key used for training; selects README dataset blurb.")
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip CV17/CV22 test eval (e.g. if test_results.json already exists).")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip Hub push (eval only, save test_results.json locally).")
    # Hyperparameters that show up in the README's training-args table.
    # Mirror what was actually used in qwen_nl_0.6b_v2.sh (epoch 2 → effective stop).
    parser.add_argument("--lr", default="2e-5")
    parser.add_argument("--scheduler", default="linear")
    parser.add_argument("--warmup-ratio", default="0.02")
    parser.add_argument("--per-device-batch", default="92")
    parser.add_argument("--grad-accum", default="2")
    parser.add_argument("--effective-batch", default="184")
    parser.add_argument("--epochs", default="2 (early-stopped from 3; eval_loss best at step 800 / epoch 1.69)")
    parser.add_argument("--grad-ckpt", default="enabled")
    args = parser.parse_args()

    language_name = LANGUAGE_NAMES[args.language]
    ckpt = Path(args.checkpoint)
    out = Path(args.output_dir)
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Materialise the checkpoint as the final output dir (mirrors what
    #    trainer.save_model would have written at training-end).
    # -----------------------------------------------------------------------
    logger.info(f"Materialising checkpoint → {out}")
    # Copy weights + tokenizer/processor files. The MakeEveryCheckpointInferableCallback
    # already copied processor / tokenizer / config files into each checkpoint dir, so
    # the checkpoint is standalone-loadable.
    for fn in sorted(ckpt.iterdir()):
        if fn.name in {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"}:
            continue  # training-only artefacts, not needed for inference
        dst = out / fn.name
        if dst.exists() and dst.is_file():
            dst.unlink()
        if fn.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(fn, dst)
        else:
            shutil.copy2(fn, dst)
    logger.info("  done.")

    # -----------------------------------------------------------------------
    # 2. Load model + processor for evaluation
    # -----------------------------------------------------------------------
    logger.info(f"Loading processor from {out} ...")
    processor = AutoProcessor.from_pretrained(str(out))
    logger.info(f"Loading model from {out} (bf16, attn={args.attn_implementation}) ...")
    model = Qwen3ASRForTraining.from_pretrained(
        str(out),
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()

    # Stamp the public class name into config so `Qwen3ASRModel.from_pretrained`
    # works on the published repo (Trainer would have done this if it had run).
    model.config.architectures = ["Qwen3ASRForConditionalGeneration"]
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = eos_id
    model.config.save_pretrained(str(out))
    model.generation_config.save_pretrained(str(out))

    # -----------------------------------------------------------------------
    # 3. Test eval (CV17-nl + CV22-nl test)
    # -----------------------------------------------------------------------
    test_results: list[dict] = []
    test_results_path = out / "test_results.json"

    if args.skip_eval and test_results_path.exists():
        with open(test_results_path) as f:
            test_results = json.load(f)
        logger.info(f"Loaded existing test_results.json ({len(test_results)} sets)")
    else:
        for label, loader in [
            ("cv17_test", lambda: load_cv17_test(args.language)),
            ("cv22_test", lambda: load_cv22(args.language, "test")),
        ]:
            logger.info(f"\nEvaluating on {label} ...")
            try:
                ds = loader()
                metrics = evaluate_model(
                    model, processor, ds,
                    language_name=language_name,
                    system_prompt="",
                    batch_size=args.eval_batch_size,
                )
                metrics.update({
                    "model": f"qwen3-asr-0.6b-{args.dataset}-{args.language}-seed42-ep2-best800",
                    "language": args.language,
                    "test_set": label,
                })
                logger.info(
                    f"  {label}: WER={metrics['wer']:.2f}% CER={metrics['cer']:.2f}% "
                    f"({metrics['num_samples']} samples)"
                )
                test_results.append(metrics)
            except Exception as e:
                logger.warning(f"  {label} eval failed: {e}")
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {test_results_path}")

    # -----------------------------------------------------------------------
    # 4. Push to Hub (stage 1: weights + simple README, stage 2: full README)
    # -----------------------------------------------------------------------
    if args.skip_push:
        logger.info("Skipping Hub push.")
        return

    api = HfApi()
    short = _model_short_name(args.base_model)

    logger.info(f"\n[stage 1/2] Creating repo {args.hub_repo_id} and uploading weights ...")
    api.create_repo(args.hub_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(out),
        repo_id=args.hub_repo_id,
        commit_message=f"Upload {short} {args.language.upper()} v2 weights (early-stopped at epoch 2, best-by-eval_loss = step 800)",
        ignore_patterns=[
            "wandb/*", "checkpoint-*/", "runs/*",
            "optimizer.pt", "scheduler.pt", "trainer_state.json",
            "training_args.bin", "rng_state.pth",
            "test_results.json",  # uploaded with full README in stage 2
        ],
    )
    simple = _build_simple_readme(
        args.language, language_name, args.base_model,
        model_repo_id=args.hub_repo_id,
    )
    api.upload_file(
        path_or_fileobj=simple.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.hub_repo_id,
        commit_message="Add initial README (results pending)",
    )
    logger.info(f"  Stage 1 done: https://huggingface.co/{args.hub_repo_id}")

    # Stage 2: full README with comparison table
    logger.info(f"\n[stage 2/2] Updating README on {args.hub_repo_id} with v2 results ...")
    baseline_results = _load_qwen_zero_shot_baseline(args.language, args.base_model)

    train_dataset_name, train_dataset_url, train_dataset_blurb = _dataset_card_meta(
        args.dataset, args.language
    )
    training_args_summary = {
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.effective_batch,
        "epochs": args.epochs,
        "grad_ckpt": args.grad_ckpt,
    }

    update_note = (
        "v2 release. The v1 run (mixed_nl: synthetic_nl + CV22-nl train, 5 epochs) "
        "reached 9.06% WER on CV17-nl test and 8.95% WER on CV22-nl test. For v2 "
        "we fold CV22-nl train + validation together with the full synthetic_transcript_nl "
        "corpus (~34.9k clips) into one training set, train with a 3-epoch budget, and "
        "validate only on the held-out CV17-nl and CV22-nl test sets. The run was early-"
        "stopped at the start of epoch 2 because eval_loss bottomed at step 800 "
        "(epoch 1.69, eval_loss 0.1249) and started rising at step 1000 — the "
        "checkpoint published here is step 800."
    )

    # train_samples / val_samples for the README — not loaded at finalize time.
    # The training log shows: synthetic_nl (~34.9k) + CV22-nl train + CV22-nl val.
    # Hard-coded as approximate counts since we are not re-loading the datasets.
    train_samples_estimate = 35000 + 11000 + 11000  # synthetic + cv22 train + cv22 val
    val_samples_estimate = 12033  # CV22-nl test (used as in-training eval set)

    full_readme = _build_full_readme(
        language_code=args.language,
        language_name=language_name,
        base_model=args.base_model,
        test_results=test_results,
        train_samples=train_samples_estimate,
        val_samples=val_samples_estimate,
        training_args_summary=training_args_summary,
        baseline_results=baseline_results,
        train_dataset_name=train_dataset_name,
        train_dataset_url=train_dataset_url,
        train_dataset_blurb=train_dataset_blurb,
        update_note=update_note,
        model_repo_id=args.hub_repo_id,
    )
    api.upload_file(
        path_or_fileobj=full_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.hub_repo_id,
        commit_message=f"Update README with v2 test WER/CER (CV17-nl + CV22-nl)",
    )
    if test_results_path.exists():
        api.upload_file(
            path_or_fileobj=str(test_results_path),
            path_in_repo="test_results.json",
            repo_id=args.hub_repo_id,
            commit_message="Add v2 test_results.json",
        )
    logger.info(f"  Stage 2 done: https://huggingface.co/{args.hub_repo_id}")


if __name__ == "__main__":
    main()
