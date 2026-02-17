"""
Fine-tune Whisper-large-v3 on CommonVoice + synthetic data for Estonian/Slovenian.

Loads dataset configurations from the HuggingFace dataset repo
(yuriyvnv/synthetic_asr_et_sl) which contains pre-built train/validation/test
splits for each experimental condition.

Usage:
    uv run python -m src.training.train_whisper \
        --language et \
        --config cv_synth_all_et \
        --output-dir ./results/whisper_et_cv_synth_all \
        --seed 42

    uv run python -m src.training.train_whisper \
        --language sl \
        --config cv_only_sl \
        --output-dir ./results/whisper_sl_cv_only \
        --seed 42
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WHISPER_MODEL_ID = "openai/whisper-large-v3"
DATASET_REPO = "yuriyvnv/synthetic_asr_et_sl"
WHISPER_LANGUAGES = {"et": "estonian", "sl": "slovenian"}

VALID_CONFIGS = [
    "cv_only_et", "cv_only_sl",
    "cv_synth_all_et", "cv_synth_all_sl",
    "cv_synth_no_morph_et", "cv_synth_no_morph_sl",
    "cv_synth_unfiltered_et", "cv_synth_unfiltered_sl",
]


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-large-v3 for Estonian/Slovenian ASR"
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
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=4000,
        help="Maximum training steps (default: 4000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Per-device train batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push final model to HuggingFace Hub",
    )
    args = parser.parse_args()

    # Setup
    os.environ.setdefault("WANDB_PROJECT", "syntts-asr-whisper")
    run_name = f"whisper-large-v3-{args.config}-seed{args.seed}"
    language = WHISPER_LANGUAGES[args.language]

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    logger.info(f"Loading dataset: {DATASET_REPO} / {args.config}")
    dataset = load_dataset(DATASET_REPO, args.config)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    logger.info(f"  Train: {len(train_ds)} samples")
    logger.info(f"  Validation: {len(val_ds)} samples")

    # -----------------------------------------------------------------------
    # Processor & preprocessing
    # -----------------------------------------------------------------------
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        WHISPER_MODEL_ID, language=language, task="transcribe"
    )

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    logger.info("Preprocessing train dataset...")
    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=train_ds.column_names,
        desc="Processing train",
    )
    logger.info("Preprocessing validation dataset...")
    val_ds = val_ds.map(
        prepare_dataset,
        remove_columns=val_ds.column_names,
        desc="Processing validation",
    )

    train_ds = train_ds.shuffle(seed=args.seed)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    logger.info("Loading Whisper-large-v3...")
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL_ID,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        seed=args.seed,

        # Batch & optimization
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        max_steps=args.max_steps,
        bf16=True,
        optim="adamw_torch_fused",

        # Evaluation on val loss only (WER evaluated separately after training)
        eval_strategy="steps",
        eval_steps=500,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Saving
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,

        # Logging
        logging_steps=25,
        report_to=["wandb"],

        # Hub
        push_to_hub=args.push_to_hub,
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=processor,
    )

    logger.info(f"Starting training: {run_name}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Seed: {args.seed}")

    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    if args.push_to_hub:
        trainer.push_to_hub()
        logger.info("Model pushed to HuggingFace Hub")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
