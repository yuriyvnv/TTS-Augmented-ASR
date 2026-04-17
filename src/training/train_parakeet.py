"""
Fine-tune Parakeet-TDT-0.6B-v3 on CommonVoice + synthetic data.

For Estonian/Slovenian: loads pre-built dataset configurations from
the HuggingFace dataset repo (yuriyvnv/synthetic_asr_et_sl).

For Dutch: loads fixie-ai/common_voice_17_0 (nl) and
yuriyvnv/synthetic_transcript_nl separately, combines at runtime.

Usage:
    uv run python -m src.training.train_parakeet \
        --language et \
        --config cv_synth_all_et \
        --output-dir ./results/parakeet_finetune_et \
        --seed 42

    uv run python -m src.training.train_parakeet \
        --language nl \
        --config cv_synth_nl \
        --output-dir ./results/parakeet_finetune_nl \
        --seed 42
"""

import argparse
import json
import logging
import os
from pathlib import Path

import lightning.pytorch as pl
import soundfile as sf
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("nemo_logger").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DATASET_REPO = "yuriyvnv/synthetic_asr_et_sl"

VALID_CONFIGS = [
    "cv_only_et", "cv_only_sl",
    "cv_synth_all_et", "cv_synth_all_sl",
    "cv_synth_no_morph_et", "cv_synth_no_morph_sl",
    "cv_synth_unfiltered_et", "cv_synth_unfiltered_sl",
    # Dutch — loaded from separate sources at runtime
    "cv_only_nl", "cv_synth_nl",
    # Portuguese — loaded from yuriyvnv/synthetic_transcript_pt
    "mixed_cv_synthetic_pt",
    # Polish — filtered BIGOS v2 (cased+punctuated sources only)
    "bigos_cased_pl",
]

# Dutch-specific dataset sources
CV17_REPO = "fixie-ai/common_voice_17_0"
SYNTH_NL_REPO = "yuriyvnv/synthetic_transcript_nl"

# Portuguese dataset
SYNTH_PT_REPO = "yuriyvnv/synthetic_transcript_pt"

# Polish dataset
BIGOS_REPO = "amu-cai/pl-asr-bigos-v2"
BIGOS_CASED_SOURCES = [
    "mozilla-common_voice_15-23",
    "mailabs-corpus_librivox-19",
    "polyai-minds14-21",
]


# ---------------------------------------------------------------------------
# Data preparation — convert HF dataset to NeMo manifests
# ---------------------------------------------------------------------------


def hf_to_nemo_manifest(
    dataset,
    split_name: str,
    audio_dir: Path,
    manifest_path: Path,
) -> str:
    """Convert a HuggingFace dataset split to NeMo JSONL manifest.

    Saves audio as 16kHz WAV files and creates a manifest with
    {audio_filepath, text, duration} entries.

    Returns the path to the manifest file.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(
            tqdm(dataset, desc=f"Converting {split_name}")
        ):
            audio = sample["audio"]
            text = sample["sentence"]

            # Save audio as WAV
            wav_path = audio_dir / f"{split_name}_{i:06d}.wav"
            sf.write(
                str(wav_path),
                audio["array"],
                audio["sampling_rate"],
                subtype="PCM_16",
            )

            duration = len(audio["array"]) / audio["sampling_rate"]

            entry = {
                "audio_filepath": str(wav_path),
                "text": text,
                "duration": round(duration, 3),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"  {split_name} manifest: {manifest_path} ({i + 1} samples)")
    return str(manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Parakeet-TDT-0.6B-v3 for Estonian/Slovenian/Dutch ASR"
    )
    parser.add_argument(
        "--language", type=str, required=True, choices=["et", "sl", "nl", "pt", "pl"],
        help="Target language",
    )
    parser.add_argument(
        "--config", type=str, required=True, choices=VALID_CONFIGS,
        help="Dataset configuration from HuggingFace repo",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for checkpoints and final model",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory for NeMo manifests and audio (default: output-dir/data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100,
        help="Maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Stop if val_wer doesn't improve for N epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Train batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="Peak learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.10,
        help="Warmup as fraction of total steps (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push final model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id", type=str, default="yuriyvnv/experiments_parakeet",
        help="HuggingFace repo ID to upload results folder (default: yuriyvnv/experiments_parakeet)",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"parakeet-tdt-{args.config}-seed{args.seed}"

    # -----------------------------------------------------------------------
    # Load dataset and convert to NeMo format
    # -----------------------------------------------------------------------
    if args.language == "nl":
        # Dutch: load CV17 and synthetic dataset separately, combine
        logger.info(f"Loading Common Voice 17 (nl) from {CV17_REPO}...")
        cv_train = load_dataset(CV17_REPO, "nl", split="train")
        cv_val = load_dataset(CV17_REPO, "nl", split="validation")
        # Standardise columns: keep audio + rename sentence
        cv_train = cv_train.select_columns(["audio", "sentence"])
        cv_val = cv_val.select_columns(["audio", "sentence"])

        # Cast audio to 16kHz before concatenation to align features
        cv_train = cv_train.cast_column("audio", Audio(sampling_rate=16000))
        cv_val = cv_val.cast_column("audio", Audio(sampling_rate=16000))

        if args.config == "cv_synth_nl":
            logger.info(f"Loading synthetic data from {SYNTH_NL_REPO}...")
            synth = load_dataset(SYNTH_NL_REPO, split="train")
            # Rename 'text' -> 'sentence' to match CV schema
            synth = synth.rename_column("text", "sentence")
            synth = synth.select_columns(["audio", "sentence"])
            synth = synth.cast_column("audio", Audio(sampling_rate=16000))
            train_ds = concatenate_datasets([cv_train, synth])
            logger.info(f"  CV17 train: {len(cv_train)}, Synthetic: {len(synth)}")
        else:
            # cv_only_nl
            train_ds = cv_train

        logger.info(f"  Train (combined): {len(train_ds)} samples")
        logger.info(f"  Validation: {len(cv_val)} samples")

        # Wrap in dict-like for uniform access below
        dataset = {"train": train_ds, "validation": cv_val}
    elif args.language == "pt":
        # Portuguese: pre-built config in yuriyvnv/synthetic_transcript_pt
        logger.info(f"Loading dataset: {SYNTH_PT_REPO} / mixed_cv_synthetic")
        pt_ds = load_dataset(SYNTH_PT_REPO, "mixed_cv_synthetic")
        # Rename 'text' -> 'sentence' for NeMo manifest compatibility
        pt_ds = pt_ds.rename_column("text", "sentence")
        pt_ds = pt_ds.select_columns(["audio", "sentence"])
        pt_ds = pt_ds.cast_column("audio", Audio(sampling_rate=16000))

        logger.info(f"  Train: {len(pt_ds['train'])} samples")
        logger.info(f"  Validation: {len(pt_ds['validation'])} samples")

        dataset = {"train": pt_ds["train"], "validation": pt_ds["validation"]}
    elif args.language == "pl":
        # Polish: BIGOS v2 filtered to cased+punctuated sources only
        logger.info(f"Loading BIGOS v2 (all) from {BIGOS_REPO}...")
        pl_train = load_dataset(BIGOS_REPO, "all", split="train", trust_remote_code=True)
        pl_val = load_dataset(BIGOS_REPO, "all", split="validation", trust_remote_code=True)

        logger.info(f"  Filtering to cased sources: {BIGOS_CASED_SOURCES}")
        pl_train = pl_train.filter(lambda x: x["dataset"] in BIGOS_CASED_SOURCES)
        pl_val = pl_val.filter(lambda x: x["dataset"] in BIGOS_CASED_SOURCES)

        # Rename ref_orig -> sentence for NeMo manifest compatibility
        pl_train = pl_train.rename_column("ref_orig", "sentence")
        pl_val = pl_val.rename_column("ref_orig", "sentence")
        pl_train = pl_train.select_columns(["audio", "sentence"])
        pl_val = pl_val.select_columns(["audio", "sentence"])
        pl_train = pl_train.cast_column("audio", Audio(sampling_rate=16000))
        pl_val = pl_val.cast_column("audio", Audio(sampling_rate=16000))

        logger.info(f"  Train (filtered): {len(pl_train)} samples")
        logger.info(f"  Validation (filtered): {len(pl_val)} samples")

        dataset = {"train": pl_train, "validation": pl_val}
    else:
        # Estonian / Slovenian: load from pre-built HF dataset
        logger.info(f"Loading dataset: {DATASET_REPO} / {args.config}")
        dataset = load_dataset(DATASET_REPO, args.config)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        logger.info(f"  Train: {len(dataset['train'])} samples")
        logger.info(f"  Validation: {len(dataset['validation'])} samples")

    # Check if manifests already exist (resume support)
    train_manifest = data_dir / "train_manifest.jsonl"
    val_manifest = data_dir / "val_manifest.jsonl"

    if train_manifest.exists() and val_manifest.exists():
        logger.info("Manifests already exist, skipping conversion")
    else:
        logger.info("Converting HF dataset to NeMo manifest format...")
        hf_to_nemo_manifest(
            dataset["train"], "train",
            data_dir / "train_audio", train_manifest,
        )
        hf_to_nemo_manifest(
            dataset["validation"], "val",
            data_dir / "val_audio", val_manifest,
        )

    # -----------------------------------------------------------------------
    # Load pre-trained model
    # -----------------------------------------------------------------------
    import nemo.collections.asr as nemo_asr

    logger.info(f"Loading Parakeet from {PARAKEET_MODEL_ID}...")
    model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL_ID)

    # Disable CUDA graph decoding — workaround for NeMo v2.2.1 bug where
    # CUDA graphs + bf16-mixed precision corrupt decoder state during
    # validation, causing repetition loops and inflated WER.
    # Fixed in NeMo v2.3.0 (PR #12938), but we're on v2.2.1.
    from omegaconf import open_dict
    with open_dict(model.cfg):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding)
    logger.info("Disabled CUDA graph decoder (NeMo v2.2.1 bf16 workaround)")

    # -----------------------------------------------------------------------
    # Configure datasets
    # -----------------------------------------------------------------------
    # Disable Lhotse (use standard NeMo dataloader with fixed batch_size)
    # Lhotse uses dynamic batching by duration which is overkill for small datasets
    train_ds_cfg = OmegaConf.create({
        "manifest_filepath": str(train_manifest),
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "max_duration": 20.0,
        "min_duration": 0.1,
    })
    model.cfg.train_ds = train_ds_cfg

    val_ds_cfg = OmegaConf.create({
        "manifest_filepath": str(val_manifest),
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": True,
        "max_duration": 40.0,
        "min_duration": 0.1,
    })
    model.cfg.validation_ds = val_ds_cfg

    # Apply dataset configs
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    # -----------------------------------------------------------------------
    # Configure optimizer
    # -----------------------------------------------------------------------
    optim_cfg = OmegaConf.create({
        "name": "adamw",
        "lr": args.learning_rate,
        "betas": [0.9, 0.98],
        "weight_decay": 0.001,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_ratio": args.warmup_ratio,
            "min_lr": 1e-6,
        },
    })
    model.cfg.optim = optim_cfg

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    # WandB logging
    os.environ.setdefault("WANDB_PROJECT", "syntts-asr-parakeet")

    wandb_logger = pl.loggers.WandbLogger(
        name=run_name,
        project=os.environ.get("WANDB_PROJECT", "syntts-asr-parakeet"),
        save_dir=str(output_dir),
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_wer",
        mode="min",
        patience=args.early_stopping_patience,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=25,
        default_root_dir=str(output_dir),
    )

    logger.info(f"Starting training: {run_name}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max epochs: {args.max_epochs}")
    logger.info(f"  Early stopping patience: {args.early_stopping_patience}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Warmup ratio: {args.warmup_ratio}")
    logger.info(f"  Seed: {args.seed}")

    trainer.fit(model)

    # Load best checkpoint weights before saving .nemo
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        logger.info(f"Loading best checkpoint: {best_ckpt}")
        checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"  Best val_wer: {checkpoint_callback.best_model_score:.4f}")

    # Save best model as .nemo
    final_path = output_dir / f"{run_name}.nemo"
    model.save_to(str(final_path))
    logger.info(f"Best model saved to {final_path}")

    # -----------------------------------------------------------------------
    # Evaluate on test sets
    # -----------------------------------------------------------------------
    import numpy as np
    from jiwer import cer as compute_cer
    from jiwer import wer as compute_wer

    def _evaluate_test(test_name, test_ds, ref_col="sentence"):
        """Run evaluation on a test set and return results dict."""
        logger.info(f"\nEvaluating best model on {test_name}...")
        refs = test_ds[ref_col]
        arrays = [
            np.array(s["audio"]["array"], dtype=np.float32)
            for s in tqdm(test_ds, desc=f"Preparing {test_name} audio")
        ]
        logger.info(f"Transcribing {len(arrays)} samples...")
        outs = model.transcribe(arrays, batch_size=args.batch_size)
        hyps = [o.text.strip() if hasattr(o, "text") else str(o).strip() for o in outs]
        w = compute_wer(refs, hyps)
        c = compute_cer(refs, hyps)
        logger.info(f"  {test_name}: WER={w * 100:.2f}%  CER={c * 100:.2f}%  ({len(refs)} samples)")
        return {"model": run_name, "language": args.language, "test_set": test_name,
                "wer": round(w * 100, 2), "cer": round(c * 100, 2), "num_samples": len(refs)}

    all_test_results = []

    # 1) Common Voice 17 test
    cv_test = load_dataset(CV17_REPO, args.language, split="test")
    cv_test = cv_test.select_columns(["audio", "sentence"])
    cv_test = cv_test.cast_column("audio", Audio(sampling_rate=16000))
    all_test_results.append(_evaluate_test("cv17_test", cv_test))

    # 2) BIGOS v2 test (Polish only — filtered to cased sources)
    if args.language == "pl":
        bigos_test = load_dataset(BIGOS_REPO, "all", split="test", trust_remote_code=True)
        bigos_test = bigos_test.filter(lambda x: x["dataset"] in BIGOS_CASED_SOURCES)
        bigos_test = bigos_test.rename_column("ref_orig", "sentence")
        bigos_test = bigos_test.select_columns(["audio", "sentence"])
        bigos_test = bigos_test.cast_column("audio", Audio(sampling_rate=16000))
        all_test_results.append(_evaluate_test("bigos_cased_test", bigos_test))

    # Save all test results
    test_results_path = output_dir / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(all_test_results, f, indent=2)
    logger.info(f"\nAll test results saved to {test_results_path}")

    # Push results folder to HuggingFace Hub
    if args.push_to_hub:
        from huggingface_hub import HfApi

        hub_repo_id = args.hub_repo_id
        logger.info(f"Pushing results folder to {hub_repo_id}...")
        api = HfApi()
        api.create_repo(hub_repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=hub_repo_id,
            path_in_repo=run_name,
            commit_message=f"Upload {run_name}",
            ignore_patterns=["data/*", "wandb/*"],
        )
        logger.info(f"Results pushed to https://huggingface.co/{hub_repo_id}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
