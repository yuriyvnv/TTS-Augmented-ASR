"""Measure zero-shot Qwen3-ASR-1.7B baseline on Portuguese test sets, using
the SAME code path as the in-training fine-tuned eval so the README comparison
is apples-to-apples.

What this fixes:
  The pre-existing baseline at results/qwenV3/pt/qwen3-asr-1.7b_pt_cv17_test.json
  (WER=13.94%) was produced by src/evaluation/evaluate.py via the qwen-asr
  package's `Qwen3ASRModel.transcribe(...)` API on RAW references (no
  normalization). The fine-tuned eval (8.40% / 8.72%) was produced by
  train_qwen3_asr.evaluate_model(...) on NORMALIZED references with raw
  model.generate(). Different protocols → comparison is misleading.

  This script re-measures the zero-shot model with the fine-tuned protocol
  on both CV17-pt test and CV22-pt test, writing:
      results/qwenV3/pt/qwen3-asr-1.7b_pt_cv17_test_baseline.json
      results/qwenV3/pt/qwen3-asr-1.7b_pt_cv22_test_baseline.json

Usage (inside Docker container, after training completes):
    bash scripts/docker/shell.sh
    cd /workspace
    uv run python scripts/evaluate/qwen_pt_zero_shot_baseline.py

Approx runtime: ~25 min on H100 for both test sets at batch=12.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.training.train_qwen3_asr import (
    QWEN_MODEL_ID,
    Qwen3ASRForTraining,
    evaluate_model,
    load_cv17_test,
    load_cv22,
)
from transformers import AutoProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="pt", choices=["pt"])
    parser.add_argument("--language-name", default="Portuguese")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Inference-only, no grad/optimizer state — H100 "
                             "has plenty of headroom. Drop to 64 if you see "
                             "OOM on long audio (>20s).")
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Use sdpa if flash-attn isn't available on the host (Docker has it).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/qwenV3/pt"),
        help="Where to write the baseline JSONs.",
    )
    parser.add_argument(
        "--test-sets", nargs="+", default=["cv17_test", "cv22_test"],
        choices=["cv17_test", "cv22_test"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base {QWEN_MODEL_ID} (bf16, attn={args.attn_implementation})...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = Qwen3ASRForTraining.from_pretrained(
        QWEN_MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()

    for ts in args.test_sets:
        logger.info(f"\n{'='*60}\nZero-shot eval on {ts} ({args.language})\n{'='*60}")
        if ts == "cv17_test":
            ds = load_cv17_test(args.language)
        elif ts == "cv22_test":
            ds = load_cv22(args.language, "test")
        else:
            raise ValueError(ts)

        metrics = evaluate_model(
            model, processor, ds,
            language_name=args.language_name,
            system_prompt="",
            batch_size=args.batch_size,
        )
        metrics.update({
            "model": f"{QWEN_MODEL_ID} (zero-shot, normalized refs)",
            "language": args.language,
            "test_set": ts,
            "protocol": "train_qwen3_asr.evaluate_model + normalize_written_form",
        })
        out_path = args.output_dir / f"qwen3-asr-1.7b_{args.language}_{ts}_baseline.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(
            f"  {ts}: WER={metrics['wer']:.2f}%  CER={metrics['cer']:.2f}%  "
            f"({metrics['num_samples']} samples) → {out_path}"
        )


if __name__ == "__main__":
    main()
