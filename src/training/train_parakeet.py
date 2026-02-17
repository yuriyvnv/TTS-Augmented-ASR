"""
Fine-tune Parakeet-TDT-0.6B-v3 on CommonVoice + synthetic data for Estonian/Slovenian.

Loads dataset configurations from the HuggingFace dataset repo
(yuriyvnv/synthetic_asr_et_sl), converts to NeMo manifest format,
and fine-tunes using PyTorch Lightning via NeMo.

Usage:
    uv run python -m src.training.train_parakeet \
        --language et \
        --config cv_synth_all_et \
        --output-dir ./results/parakeet_finetune_et \
        --seed 42

    uv run python -m src.training.train_parakeet \
        --language sl \
        --config cv_only_sl \
        --output-dir ./results/parakeet_finetune_sl \
        --seed 42
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
from datasets import Audio, load_dataset
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
        description="Fine-tune Parakeet-TDT-0.6B-v3 for Estonian/Slovenian ASR"
    )
    parser.add_argument(
        "--language", type=str, required=True, choices=["et", "sl"],
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

    # Save final model as .nemo
    final_path = output_dir / f"{run_name}.nemo"
    model.save_to(str(final_path))
    logger.info(f"Final model saved to {final_path}")

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
            ignore_patterns=["data/*"],
        )
        logger.info(f"Results pushed to https://huggingface.co/{hub_repo_id}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
