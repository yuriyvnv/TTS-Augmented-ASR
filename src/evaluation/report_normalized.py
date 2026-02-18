"""
Report raw vs normalized WER/CER from existing evaluation JSON files.

Reads per-sentence results from results/*.json, recomputes metrics with
jiwer's built-in text transforms (lowercase + punctuation removal), and
outputs a comparison table.

Usage:
    uv run python -m src.evaluation.report_normalized
    uv run python -m src.evaluation.report_normalized --save-xlsx
    uv run python -m src.evaluation.report_normalized --results-dir ./results
"""

import argparse
import json
import logging
from pathlib import Path

import jiwer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# ---------------------------------------------------------------------------
# jiwer transforms for normalized WER/CER
# ---------------------------------------------------------------------------

# Normalized WER: lowercase, remove punctuation, collapse spaces
NORM_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

# Normalized CER: lowercase, remove punctuation
NORM_CER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfChars(),
])


# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

def parse_result_filename(filename: str) -> dict | None:
    """Parse a result JSON filename into model, lang, test_set.

    Expected pattern: {model}_{lang}_{test_set}.json
    where lang is 'et' or 'sl' and test_set is 'cv17_test', etc.
    """
    stem = filename.replace(".json", "")

    # Try matching known test set suffixes
    for test_set in ["cv17_validation", "cv17_test", "fleurs_test"]:
        if stem.endswith(f"_{test_set}"):
            prefix = stem[: -len(f"_{test_set}")]
            # Extract language (last 2 chars of prefix after the last _)
            parts = prefix.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("et", "sl"):
                return {
                    "model": parts[0],
                    "lang": parts[1],
                    "test_set": test_set,
                }
    return None


def load_results(results_dir: Path) -> list[dict]:
    """Load all evaluation JSON files and parse metadata from filenames."""
    entries = []
    for path in sorted(results_dir.glob("**/*.json")):
        meta = parse_result_filename(path.name)
        if meta is None:
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        refs = [s["reference"] for s in data["per_sentence"]]
        hyps = [s["hypothesis"] for s in data["per_sentence"]]

        entries.append({
            "path": path,
            "model": meta["model"],
            "lang": meta["lang"],
            "test_set": meta["test_set"],
            "num_samples": len(refs),
            "raw_wer": data["wer"],
            "raw_cer": data["cer"],
            "refs": refs,
            "hyps": hyps,
        })

    return entries


# ---------------------------------------------------------------------------
# Compute normalized metrics
# ---------------------------------------------------------------------------

def compute_normalized(refs: list[str], hyps: list[str]) -> dict:
    """Compute normalized WER and CER using jiwer transforms."""
    norm_wer = jiwer.wer(
        refs, hyps,
        reference_transform=NORM_WER_TRANSFORM,
        hypothesis_transform=NORM_WER_TRANSFORM,
    )
    norm_cer = jiwer.cer(
        refs, hyps,
        reference_transform=NORM_CER_TRANSFORM,
        hypothesis_transform=NORM_CER_TRANSFORM,
    )
    return {
        "norm_wer": round(norm_wer * 100, 2),
        "norm_cer": round(norm_cer * 100, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Report raw vs normalized WER/CER from evaluation results"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--save-xlsx", action="store_true",
        help="Save results to Excel file",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    logger.info(f"Loading results from {results_dir}...")
    entries = load_results(results_dir)
    logger.info(f"  Found {len(entries)} result files")

    # Compute normalized metrics for each entry
    rows = []
    for e in entries:
        logger.info(f"  Computing normalized metrics: {e['model']} / {e['lang']} / {e['test_set']}...")
        norm = compute_normalized(e["refs"], e["hyps"])
        rows.append({
            "Model": e["model"],
            "Lang": e["lang"],
            "Test Set": e["test_set"],
            "Samples": e["num_samples"],
            "Raw WER": e["raw_wer"],
            "Norm WER": norm["norm_wer"],
            "WER Diff": round(norm["norm_wer"] - e["raw_wer"], 2),
            "Raw CER": e["raw_cer"],
            "Norm CER": norm["norm_cer"],
            "CER Diff": round(norm["norm_cer"] - e["raw_cer"], 2),
        })

    # Sort by lang, then model, then test_set
    rows.sort(key=lambda r: (r["Lang"], r["Model"], r["Test Set"]))

    # Print table
    print()
    print("=" * 120)
    print("  Raw vs Normalized WER/CER (lowercase + punctuation removal)")
    print("=" * 120)
    header = (
        f"  {'Model':<40} {'Lang':<5} {'Test Set':<16} {'Samples':>7}  "
        f"{'Raw WER':>8} {'Norm WER':>9} {'Diff':>6}  "
        f"{'Raw CER':>8} {'Norm CER':>9} {'Diff':>6}"
    )
    print(header)
    print(f"  {'-'*40} {'-'*5} {'-'*16} {'-'*7}  {'-'*8} {'-'*9} {'-'*6}  {'-'*8} {'-'*9} {'-'*6}")

    current_lang = None
    for r in rows:
        if r["Lang"] != current_lang:
            if current_lang is not None:
                print()
            current_lang = r["Lang"]

        print(
            f"  {r['Model']:<40} {r['Lang']:<5} {r['Test Set']:<16} {r['Samples']:>7}  "
            f"{r['Raw WER']:>7.2f}% {r['Norm WER']:>8.2f}% {r['WER Diff']:>+5.2f}  "
            f"{r['Raw CER']:>7.2f}% {r['Norm CER']:>8.2f}% {r['CER Diff']:>+5.2f}"
        )

    print()
    print("=" * 120)
    print("  Normalization: ToLowerCase + RemovePunctuation (jiwer built-in transforms)")
    print("  Diff = Normalized - Raw (negative means normalization reduces error rate)")
    print("=" * 120)
    print()

    # Save to Excel
    if args.save_xlsx:
        import pandas as pd

        df = pd.DataFrame(rows)
        out_path = results_dir / "normalized_results.xlsx"
        df.to_excel(out_path, index=False, sheet_name="Raw vs Normalized")
        logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
