"""
Bootstrap significance testing for ASR evaluation results.

Computes paired and unpaired bootstrap confidence intervals and p-values
for WER differences between systems.

Usage:
    uv run python -m src.evaluation.significance --language et

Reads per-sentence results from results/*.json files and produces a
summary table with significance indicators.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from jiwer import process_words

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# ---------------------------------------------------------------------------
# Result file mapping
# ---------------------------------------------------------------------------

MODELS_ET = {
    "zero-shot": "parakeet-tdt-0.6b-v3",
    "cv_only": "parakeet-tdt-cv_only_et-seed42",
    "cv_synth_all": "parakeet-tdt-cv_synth_all_et-seed42",
}

TEST_SETS = ["cv17_test", "fleurs_test"]


def load_per_sentence(results_dir: Path, model_key: str, lang: str, test_set: str) -> dict:
    """Load a result JSON and return the full dict."""
    path = results_dir / f"{model_key}_{lang}_{test_set}.json"
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_word_errors_per_sentence(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-sentence word error counts and reference word counts.

    Returns (errors, ref_counts) arrays for corpus-level bootstrap.
    We recompute from reference/hypothesis to get exact error counts
    rather than relying on the rounded per-sentence WER percentages.
    """
    sentences = data["per_sentence"]
    errors = []
    ref_counts = []

    for s in sentences:
        ref = s["reference"].strip()
        hyp = s["hypothesis"].strip()

        if ref == "":
            ref_words = 0
            err = 0 if hyp == "" else 1
        else:
            out = process_words(ref, hyp)
            err = out.substitutions + out.insertions + out.deletions
            ref_words = out.substitutions + out.deletions + out.hits

        errors.append(err)
        ref_counts.append(ref_words)

    return np.array(errors), np.array(ref_counts)


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------


def paired_bootstrap(
    errors_a: np.ndarray,
    ref_counts_a: np.ndarray,
    errors_b: np.ndarray,
    ref_counts_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is system B better than system A?

    Both systems are evaluated on the same sentences. We resample sentence
    indices and compute corpus-level WER for each system on each resample.

    Returns dict with delta_wer, ci_95, p_value.
    """
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    assert n == len(errors_b), "Systems must have same number of sentences"

    # Observed corpus-level WER
    wer_a = errors_a.sum() / ref_counts_a.sum() * 100
    wer_b = errors_b.sum() / ref_counts_b.sum() * 100
    observed_delta = wer_b - wer_a  # negative means B is better

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_wer_a = errors_a[idx].sum() / ref_counts_a[idx].sum() * 100
        boot_wer_b = errors_b[idx].sum() / ref_counts_b[idx].sum() * 100
        deltas.append(boot_wer_b - boot_wer_a)

    deltas = np.array(deltas)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)

    # One-sided p-value: proportion of bootstrap samples where
    # the sign of delta flips (i.e., delta >= 0 when observed < 0)
    if observed_delta < 0:
        p_value = np.mean(deltas >= 0)
    elif observed_delta > 0:
        p_value = np.mean(deltas <= 0)
    else:
        p_value = 1.0

    return {
        "wer_a": round(wer_a, 2),
        "wer_b": round(wer_b, 2),
        "delta_wer": round(observed_delta, 2),
        "ci_95": (round(ci_lower, 2), round(ci_upper, 2)),
        "p_value": round(p_value, 4),
    }


def unpaired_bootstrap(
    errors_a: np.ndarray,
    ref_counts_a: np.ndarray,
    errors_b: np.ndarray,
    ref_counts_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Unpaired bootstrap test: is system B better than system A?

    Resamples each system independently (as if they were evaluated
    on different data). More conservative than paired bootstrap.

    Returns dict with delta_wer, ci_95, p_value.
    """
    rng = np.random.RandomState(seed)
    n_a = len(errors_a)
    n_b = len(errors_b)

    wer_a = errors_a.sum() / ref_counts_a.sum() * 100
    wer_b = errors_b.sum() / ref_counts_b.sum() * 100
    observed_delta = wer_b - wer_a

    deltas = []
    for _ in range(n_bootstrap):
        idx_a = rng.randint(0, n_a, size=n_a)
        idx_b = rng.randint(0, n_b, size=n_b)
        boot_wer_a = errors_a[idx_a].sum() / ref_counts_a[idx_a].sum() * 100
        boot_wer_b = errors_b[idx_b].sum() / ref_counts_b[idx_b].sum() * 100
        deltas.append(boot_wer_b - boot_wer_a)

    deltas = np.array(deltas)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)

    if observed_delta < 0:
        p_value = np.mean(deltas >= 0)
    elif observed_delta > 0:
        p_value = np.mean(deltas <= 0)
    else:
        p_value = 1.0

    return {
        "wer_a": round(wer_a, 2),
        "wer_b": round(wer_b, 2),
        "delta_wer": round(observed_delta, 2),
        "ci_95": (round(ci_lower, 2), round(ci_upper, 2)),
        "p_value": round(p_value, 4),
    }


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap significance testing for ASR results"
    )
    parser.add_argument(
        "--language", type=str, default="et", choices=["et", "sl"],
        help="Language to test (default: et)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    lang = args.language
    models = MODELS_ET if lang == "et" else MODELS_ET  # extend for sl later

    # Comparisons to test
    comparisons = [
        ("zero-shot", "cv_only", "Zero-shot vs CV-only"),
        ("zero-shot", "cv_synth_all", "Zero-shot vs CV+Synth"),
        ("cv_only", "cv_synth_all", "CV-only vs CV+Synth"),
    ]

    # Preload all per-sentence data
    logger.info("Loading per-sentence results...")
    data = {}
    for model_name, model_key in models.items():
        for test_set in TEST_SETS:
            try:
                d = load_per_sentence(results_dir, model_key, lang, test_set)
                logger.info(f"  Computing word errors: {model_name} / {test_set}...")
                errors, ref_counts = get_word_errors_per_sentence(d)
                data[(model_name, test_set)] = (errors, ref_counts)
            except FileNotFoundError as e:
                logger.warning(f"  Skipping: {e}")

    # Run bootstrap tests
    print()
    print("=" * 90)
    print(f"  Bootstrap Significance Tests — Estonian (n_bootstrap={args.n_bootstrap})")
    print("=" * 90)

    for test_set in TEST_SETS:
        print(f"\n  Test set: {test_set}")
        print(f"  {'-'*84}")
        print(f"  {'Comparison':<28} {'WER_A':>7} {'WER_B':>7} {'Delta':>7}  {'Paired p':>10} {'':>5}  {'Unpaired p':>10} {'':>5}")
        print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7}  {'-'*10} {'-'*5}  {'-'*10} {'-'*5}")

        for model_a, model_b, label in comparisons:
            if (model_a, test_set) not in data or (model_b, test_set) not in data:
                print(f"  {label:<28} {'(missing data)':>40}")
                continue

            errors_a, ref_a = data[(model_a, test_set)]
            errors_b, ref_b = data[(model_b, test_set)]

            paired = paired_bootstrap(
                errors_a, ref_a, errors_b, ref_b,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
            unpaired = unpaired_bootstrap(
                errors_a, ref_a, errors_b, ref_b,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )

            print(
                f"  {label:<28} {paired['wer_a']:>6.2f}% {paired['wer_b']:>6.2f}% "
                f"{paired['delta_wer']:>+6.2f}%  "
                f"p={paired['p_value']:<8.4f} {sig_stars(paired['p_value']):>5}  "
                f"p={unpaired['p_value']:<8.4f} {sig_stars(unpaired['p_value']):>5}"
            )

            # Print CIs
            print(
                f"  {'':28} {'':7} {'':7} {'':7}  "
                f"CI=[{paired['ci_95'][0]:+.2f},{paired['ci_95'][1]:+.2f}]"
                f"        "
                f"CI=[{unpaired['ci_95'][0]:+.2f},{unpaired['ci_95'][1]:+.2f}]"
            )

    print()
    print("=" * 90)
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("  Delta = WER_B - WER_A (negative means B is better)")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
