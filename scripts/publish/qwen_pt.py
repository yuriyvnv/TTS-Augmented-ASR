"""
Publish the fine-tuned Qwen3-ASR-1.7B Portuguese model to its public HF repo.

Uploads weights + processor files + README with eval metrics. Idempotent —
re-running updates the README without re-uploading weights.

Usage:
    uv run python scripts/publish/qwen_pt.py
    uv run python scripts/publish/qwen_pt.py --model-dir ./results/qwen3_finetune_pt/cv22_s42
    uv run python scripts/publish/qwen_pt.py --readme-only   # skip weights, just refresh README
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from src.training.train_qwen3_asr import (
    _build_full_readme,
    _dataset_card_meta,
    _load_qwen_zero_shot_baseline,
)

load_dotenv()

REPO_ID = "yuriyvnv/Qwen3-ASR-1.7B-PT"
BASE_MODEL = "Qwen/Qwen3-ASR-1.7B"
LANGUAGE_CODE = "pt"
LANGUAGE_NAME = "Portuguese"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("results/qwen3_finetune_pt/synthetic_pt_high_quality_s42"),
        help="Directory containing the fine-tuned model (config.json, safetensors, tokenizer, etc.)",
    )
    parser.add_argument(
        "--dataset", default="synthetic_pt_high_quality",
        choices=["cv22", "synthetic_pt_high_quality"],
        help="Dataset key the run was trained on (controls README dataset blurb).",
    )
    parser.add_argument("--lr", default="2e-05")
    parser.add_argument("--scheduler", default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--per-device-batch", type=int, default=92)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--grad-ckpt", default="enabled", choices=["enabled", "disabled"])
    parser.add_argument("--train-samples", type=int, default=29267,
                        help="Number of training samples after filtering")
    parser.add_argument("--val-samples", type=int, default=9464,
                        help="Number of validation samples (CV17-pt val fallback)")
    parser.add_argument(
        "--repo-id", default=REPO_ID,
        help=f"HuggingFace repo to push to (default: {REPO_ID})",
    )
    parser.add_argument(
        "--readme-only", action="store_true",
        help="Only update the README, don't upload weights",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the README and print it, but do not push",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"Model directory not found: {args.model_dir}")

    # Load test results
    test_results_path = args.model_dir / "test_results.json"
    if test_results_path.exists():
        with open(test_results_path) as f:
            test_results = json.load(f)
        print(f"Loaded {len(test_results)} test results from {test_results_path}")
    else:
        print(f"WARN: {test_results_path} not found — README will have empty results table")
        test_results = []

    # Load zero-shot baseline (results/qwenV3/<lang>/) so the README can show
    # before/after numbers. Silently empty if not present.
    baseline_results = _load_qwen_zero_shot_baseline(LANGUAGE_CODE)
    if baseline_results:
        print(
            f"Loaded {len(baseline_results)} zero-shot baseline result(s): "
            + ", ".join(f"{b['test_set']}={b['wer']}%" for b in baseline_results)
        )
    else:
        print("WARN: no zero-shot baseline found — README will only show fine-tuned numbers")

    eff_batch = args.per_device_batch * args.grad_accum
    training_args_summary = {
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": eff_batch,
        "epochs": args.epochs,
        "grad_ckpt": args.grad_ckpt,
    }

    train_dataset_name, train_dataset_url, train_dataset_blurb = _dataset_card_meta(
        args.dataset, LANGUAGE_CODE
    )

    readme = _build_full_readme(
        language_code=LANGUAGE_CODE,
        language_name=LANGUAGE_NAME,
        base_model=BASE_MODEL,
        test_results=test_results,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        training_args_summary=training_args_summary,
        baseline_results=baseline_results,
        train_dataset_name=train_dataset_name,
        train_dataset_url=train_dataset_url,
        train_dataset_blurb=train_dataset_blurb,
    )

    if args.dry_run:
        print("=" * 80)
        print(readme)
        print("=" * 80)
        print(f"(Dry run — not pushing to {args.repo_id})")
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)

    if not args.readme_only:
        print(f"Uploading weights from {args.model_dir} to {args.repo_id} ...")
        api.upload_folder(
            folder_path=str(args.model_dir),
            repo_id=args.repo_id,
            commit_message=f"Upload fine-tuned Qwen3-ASR-1.7B {LANGUAGE_NAME} weights",
            ignore_patterns=[
                "wandb/*", "checkpoint-*/", "runs/*",
                "optimizer.pt", "scheduler.pt", "trainer_state.json",
                "training_args.bin", "rng_state.pth",
                "README.md",  # we upload the generated one separately below
            ],
        )

    print(f"Uploading README to {args.repo_id} ...")
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        commit_message="Update README with evaluation results",
    )

    if test_results_path.exists():
        print(f"Uploading test_results.json to {args.repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(test_results_path),
            path_in_repo="test_results.json",
            repo_id=args.repo_id,
            commit_message="Add test_results.json",
        )

    print(f"Done → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
