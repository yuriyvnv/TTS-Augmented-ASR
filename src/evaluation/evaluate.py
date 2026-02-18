"""
Evaluate ASR models on CommonVoice 17 and FLEURS test sets.

Supports zero-shot and fine-tuned evaluation for:
  - openai/whisper-large-v3 (encoder-decoder, transformers)
  - nvidia/parakeet-tdt-0.6b-v3 (FastConformer+TDT, NeMo)

Test sets:
  - cv17_validation: CommonVoice 17 validation split
  - cv17_test:       CommonVoice 17 test split
  - fleurs_test:     FLEURS test split

WER and CER are computed on raw text (no normalization).
Per-sentence results are saved for downstream statistical testing.

Usage:
    uv run python -m src.evaluation.evaluate \
        --model whisper-large-v3 --language et \
        --test-sets cv17_validation cv17_test fleurs_test

    uv run python -m src.evaluation.evaluate \
        --model parakeet-tdt-0.6b-v3 --language sl \
        --test-sets cv17_test fleurs_test

    uv run python -m src.evaluation.evaluate \
        --model whisper-large-v3 --language et \
        --model-path ./results/whisper_et_best \
        --test-sets cv17_test fleurs_test
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from jiwer import cer, process_words, wer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _set_low_priority():
    """Lower process priority so evaluation yields resources to training."""
    # Set nice value to 19 (lowest priority) on Unix
    try:
        os.nice(19)
        logger.info("Process priority set to nice=19 (low)")
    except (OSError, AttributeError):
        pass  # Windows or permission denied

    # Limit CUDA memory so training gets priority
    if torch.cuda.is_available():
        # Reserve at most 30% of GPU memory for evaluation
        total_mem = torch.cuda.get_device_properties(0).total_memory
        fraction = 0.30
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        logger.info(
            f"CUDA memory limited to {fraction*100:.0f}% "
            f"({total_mem * fraction / 1e9:.1f} GB / {total_mem / 1e9:.1f} GB)"
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CV17_REPO = "fixie-ai/common_voice_17_0"
FLEURS_REPO = "google/fleurs"
FLEURS_CONFIGS = {"et": "et_ee", "sl": "sl_si"}

WHISPER_MODEL_ID = "openai/whisper-large-v3"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

WHISPER_LANGUAGES = {"et": "estonian", "sl": "slovenian"}

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

VALID_TEST_SETS = ["cv17_validation", "cv17_test", "fleurs_test"]


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Pick the best available device and matching dtype.

    Returns (device_string, torch_dtype):
      - CUDA:  ("cuda:0", float16)
      - MPS:   ("mps",    float32)  — MPS does not support float16 for all ops
      - CPU:   ("cpu",    float32)
    """
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_set(test_set: str, language: str):
    """Load a test set and cast audio to 16kHz.

    Returns a HuggingFace Dataset with 'audio' (16kHz) and 'reference' columns.
    """
    if test_set == "cv17_validation":
        logger.info(f"Loading CV17 validation ({language})...")
        ds = load_dataset(CV17_REPO, language, split="validation")
        ds = ds.rename_column("sentence", "reference")
    elif test_set == "cv17_test":
        logger.info(f"Loading CV17 test ({language})...")
        ds = load_dataset(CV17_REPO, language, split="test")
        ds = ds.rename_column("sentence", "reference")
    elif test_set == "fleurs_test":
        fleurs_config = FLEURS_CONFIGS[language]
        logger.info(f"Loading FLEURS test ({fleurs_config})...")
        # FLEURS uses a legacy loading script incompatible with datasets>=3.0.
        # Load from the auto-converted Parquet branch instead.
        ds = load_dataset(
            FLEURS_REPO, fleurs_config, split="test",
            revision="refs/convert/parquet",
        )
        ds = ds.rename_column("transcription", "reference")
    else:
        raise ValueError(f"Unknown test set: {test_set}")

    # Cast audio to 16kHz mono for all models
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Keep only what we need
    ds = ds.select_columns(["audio", "reference"])

    logger.info(f"  {len(ds)} samples loaded")
    return ds


# ---------------------------------------------------------------------------
# Model evaluators
# ---------------------------------------------------------------------------


class WhisperEvaluator:
    """Whisper-large-v3 evaluation via transformers pipeline."""

    def __init__(self, language: str, model_path: str | None = None):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_id = model_path or WHISPER_MODEL_ID
        device, dtype = get_device_and_dtype()

        logger.info(f"Loading Whisper from {model_id} on {device} ({dtype})...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(
            model_path or WHISPER_MODEL_ID
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )
        self.language = WHISPER_LANGUAGES[language]
        logger.info("  Whisper ready")

    def transcribe(self, dataset, batch_size: int = 16) -> list[str]:
        """Transcribe all samples in a dataset. Returns list of hypothesis strings."""
        generate_kwargs = {
            "language": self.language,
            "task": "transcribe",
        }

        hypotheses = []
        for i in tqdm(range(0, len(dataset), batch_size), desc="Whisper transcribe"):
            batch = dataset[i : i + batch_size]
            audio_inputs = [
                {"array": a["array"], "sampling_rate": a["sampling_rate"]}
                for a in batch["audio"]
            ]
            results = self.pipe(
                audio_inputs,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
            )
            hypotheses.extend([r["text"].strip() for r in results])

        return hypotheses


class ParakeetEvaluator:
    """Parakeet-TDT-0.6B-v3 evaluation via NeMo."""

    def __init__(self, language: str, model_path: str | None = None):
        import nemo.collections.asr as nemo_asr

        device, _ = get_device_and_dtype()

        if model_path:
            logger.info(f"Loading Parakeet from {model_path}...")
            self.model = nemo_asr.models.ASRModel.restore_from(model_path)
        else:
            logger.info(f"Loading Parakeet from {PARAKEET_MODEL_ID}...")
            self.model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL_ID)

        if device.startswith("cuda"):
            self.model = self.model.cuda()
        # NeMo does not support MPS — stays on CPU for Mac
        self.model.eval()
        logger.info(f"  Parakeet ready on {'cuda' if device.startswith('cuda') else 'cpu'}")

    def transcribe(self, dataset, batch_size: int = 16) -> list[str]:
        """Transcribe all samples in a dataset. Returns list of hypothesis strings."""
        import numpy as np

        # Collect all audio arrays (already 16kHz from cast_column)
        audio_arrays = [
            np.array(sample["audio"]["array"], dtype=np.float32)
            for sample in tqdm(dataset, desc="Preparing audio")
        ]

        logger.info(f"Transcribing {len(audio_arrays)} samples...")
        outputs = self.model.transcribe(audio_arrays, batch_size=batch_size)

        hypotheses = []
        for o in outputs:
            text = o.text if hasattr(o, "text") else str(o)
            hypotheses.append(text.strip())

        return hypotheses


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Compute WER, CER, and per-sentence results on raw text."""
    # Strip whitespace only
    refs = [r.strip() for r in references]
    hyps = [h.strip() for h in hypotheses]

    # Aggregate metrics
    overall_wer = wer(refs, hyps)
    overall_cer = cer(refs, hyps)

    # Detailed word-level alignment for Sub/Ins/Del
    word_output = process_words(refs, hyps)
    total_sub = word_output.substitutions
    total_ins = word_output.insertions
    total_del = word_output.deletions
    total_hits = word_output.hits
    total_ref_words = total_sub + total_del + total_hits

    # Per-sentence WER
    per_sentence = []
    for ref, hyp in zip(refs, hyps):
        if ref == "":
            sent_wer = 0.0 if hyp == "" else 1.0
        else:
            sent_wer = wer(ref, hyp)
        per_sentence.append({
            "reference": ref,
            "hypothesis": hyp,
            "wer": round(sent_wer * 100, 2),
        })

    return {
        "wer": round(overall_wer * 100, 2),
        "cer": round(overall_cer * 100, 2),
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
        "hits": total_hits,
        "ref_word_count": total_ref_words,
        "per_sentence": per_sentence,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _set_low_priority()

    parser = argparse.ArgumentParser(
        description="Evaluate ASR models on CV17 and FLEURS test sets"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["whisper-large-v3", "parakeet-tdt-0.6b-v3"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint (omit for zero-shot)",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["et", "sl", "both"],
        help="Language to evaluate",
    )
    parser.add_argument(
        "--test-sets",
        type=str,
        nargs="+",
        required=True,
        choices=VALID_TEST_SETS,
        help="Test sets to evaluate on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for results (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    langs = ["et", "sl"] if args.language == "both" else [args.language]

    model_label = args.model
    if args.model_path:
        model_label = Path(args.model_path).stem

    # Parakeet auto-detects language, so load once for all languages
    # Whisper needs language token, so reload per language
    evaluator = None

    all_results = {}

    for lang in langs:
        if args.model == "whisper-large-v3":
            evaluator = WhisperEvaluator(lang, args.model_path)
        elif evaluator is None:
            evaluator = ParakeetEvaluator(lang, args.model_path)

        for test_set in args.test_sets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_label} on {test_set} ({lang})")
            logger.info(f"{'='*60}")

            # Load data
            ds = load_test_set(test_set, lang)

            # Transcribe
            t0 = time.time()
            hypotheses = evaluator.transcribe(ds, batch_size=args.batch_size)
            elapsed = time.time() - t0

            references = ds["reference"]

            # Compute metrics
            metrics = compute_metrics(references, hypotheses)
            metrics["num_samples"] = len(ds)
            metrics["inference_time_seconds"] = round(elapsed, 1)
            metrics["model"] = args.model
            metrics["model_path"] = args.model_path
            metrics["language"] = lang
            metrics["test_set"] = test_set
            metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Print summary
            logger.info(
                f"\n  {test_set}: WER={metrics['wer']:.2f}%  CER={metrics['cer']:.2f}%  "
                f"({metrics['num_samples']} samples, {elapsed:.1f}s)"
            )
            logger.info(
                f"  Sub={metrics['substitutions']}  Ins={metrics['insertions']}  "
                f"Del={metrics['deletions']}  Hits={metrics['hits']}"
            )

            # Save results
            result_path = output_dir / f"{model_label}_{lang}_{test_set}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"  Results saved to {result_path}")

            result_key = f"{lang}/{test_set}"
            all_results[result_key] = {
                "wer": metrics["wer"],
                "cer": metrics["cer"],
                "samples": metrics["num_samples"],
            }

    # Final summary table
    logger.info(f"\n{'='*60}")
    logger.info(f"Summary: {model_label}")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Lang/Test Set':<25} {'WER%':>8} {'CER%':>8} {'Samples':>8}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for key, r in all_results.items():
        logger.info(
            f"  {key:<25} {r['wer']:>8.2f} {r['cer']:>8.2f} {r['samples']:>8}"
        )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
