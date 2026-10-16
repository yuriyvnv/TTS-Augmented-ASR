"""
Paired bootstrap significance testing for ASR evaluation results.

Compares the five adaptation states (zero-shot + four fine-tuning configs)
of one model on CV17 test and FLEURS test, on raw and/or normalized WER.
Paired bootstrap resampling over sentences (Bisani & Ney, 2004); the
one-sided p-value is the fraction of resamples where the sign of the WER
difference reverses.

Reads the per-sentence JSON files written by src.evaluation.evaluate:
    {model_key}_{lang}_{test_set}.json
where model_key is the zero-shot model name (e.g. whisper-large-v3) or the
fine-tuned run name (e.g. whisper-large-v3-cv_synth_all_et-seed42).

Usage:
    uv run python -m src.evaluation.significance --model parakeet-tdt-0.6b-v3 --language both
    uv run python -m src.evaluation.significance --model whisper-large-v3 --language sl \
        --normalization normalized
"""

import argparse
import json
import logging
from pathlib import Path

import jiwer
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

TEST_SETS = ["cv17_test", "fleurs_test"]

# Zero-shot model name -> prefix of fine-tuned run names
RUN_PREFIXES = {
    "whisper-large-v3": "whisper-large-v3",
    "parakeet-tdt-0.6b-v3": "parakeet-tdt",
}

CONFIGS = ["cv_only", "cv_synth_no_morph", "cv_synth_all", "cv_synth_unfiltered"]

COMPARISONS = [
    ("zero-shot", "cv_only", "Zero-shot vs CV-only"),
    ("zero-shot", "cv_synth_no_morph", "Zero-shot vs CV+Synth No Morph"),
    ("zero-shot", "cv_synth_all", "Zero-shot vs CV+Synth All"),
    ("zero-shot", "cv_synth_unfiltered", "Zero-shot vs CV+Synth Unfiltered"),
    ("cv_only", "cv_synth_no_morph", "CV-only vs CV+Synth No Morph"),
    ("cv_only", "cv_synth_all", "CV-only vs CV+Synth All"),
    ("cv_only", "cv_synth_unfiltered", "CV-only vs CV+Synth Unfiltered"),
    ("cv_synth_no_morph", "cv_synth_all", "No Morph vs Synth All"),
    ("cv_synth_all", "cv_synth_unfiltered", "Synth All vs Unfiltered"),
]

# Same normalization as report_normalized.py (lowercase + punctuation removal)
NORM_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def model_keys(model: str, lang: str, run_seed: int) -> dict[str, str]:
    """Map adaptation state -> result file prefix."""
    prefix = RUN_PREFIXES[model]
    keys = {"zero-shot": model}
    for config in CONFIGS:
        keys[config] = f"{prefix}-{config}_{lang}-seed{run_seed}"
    return keys


def load_result(results_dir: Path, model_key: str, lang: str, test_set: str) -> dict | None:
    """Find {model_key}_{lang}_{test_set}.json anywhere under results_dir."""
    matches = list(results_dir.glob(f"**/{model_key}_{lang}_{test_set}.json"))
    if not matches:
        return None
    with open(matches[0], "r", encoding="utf-8") as f:
        return json.load(f)


def get_word_errors(data: dict, normalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-sentence word error counts and reference word counts.

    Recomputed from reference/hypothesis for exact counts rather than
    relying on the rounded per-sentence WER percentages.
    """
    errors, ref_counts = [], []
    for s in data["per_sentence"]:
        ref = s["reference"].strip()
        hyp = s["hypothesis"].strip()
        if normalize:
            ref = NORM_TRANSFORM(ref)
            hyp = NORM_TRANSFORM(hyp)
        if ref == "":
            errors.append(0 if hyp == "" else 1)
            ref_counts.append(0)
        else:
            out = jiwer.process_words(ref, hyp)
            errors.append(out.substitutions + out.insertions + out.deletions)
            ref_counts.append(out.substitutions + out.deletions + out.hits)
    return np.array(errors), np.array(ref_counts)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(
    errors_a: np.ndarray,
    ref_counts_a: np.ndarray,
    errors_b: np.ndarray,
    ref_counts_b: np.ndarray,
    n_bootstrap: int = 100_000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is system B better than system A?

    Resamples sentence indices and computes corpus-level WER for both
    systems on each resample. Delta = WER_B - WER_A (negative: B is better).
    """
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    assert n == len(errors_b), "Systems must have same number of sentences"

    wer_a = errors_a.sum() / ref_counts_a.sum() * 100
    wer_b = errors_b.sum() / ref_counts_b.sum() * 100
    observed_delta = wer_b - wer_a

    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = (
            errors_b[idx].sum() / ref_counts_b[idx].sum() * 100
            - errors_a[idx].sum() / ref_counts_a[idx].sum() * 100
        )

    # One-sided p-value: fraction of resamples where the sign flips
    if observed_delta < 0:
        p_value = np.mean(deltas >= 0)
    elif observed_delta > 0:
        p_value = np.mean(deltas <= 0)
    else:
        p_value = 1.0

    return {
        "wer_a": round(wer_a, 2),
        "wer_b": round(wer_b, 2),
        "delta": round(observed_delta, 2),
        "ci_95": (round(np.percentile(deltas, 2.5), 2), round(np.percentile(deltas, 97.5), 2)),
        "p_value": p_value,
    }


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_tests(
    results_dir: Path,
    model: str,
    lang: str,
    normalize: bool,
    run_seed: int,
    n_bootstrap: int,
    seed: int,
):
    norm_label = "NORMALIZED" if normalize else "RAW"
    print(f"\n{'='*100}")
    print(f"  {model} {lang.upper()} — {norm_label} WER (paired bootstrap, n={n_bootstrap})")
    print(f"{'='*100}")

    data = {}
    for state, key in model_keys(model, lang, run_seed).items():
        for ts in TEST_SETS:
            d = load_result(results_dir, key, lang, ts)
            if d is None:
                print(f"  MISSING: {key}_{lang}_{ts}.json")
                continue
            e, r = get_word_errors(d, normalize=normalize)
            data[(state, ts)] = (e, r)
            print(f"  {state}/{ts}: WER={e.sum()/r.sum()*100:.2f}% ({len(e)} sent)")

    for ts in TEST_SETS:
        print(f"\n--- {ts} ---")
        print(f"  {'Comparison':<38} {'WER_A':>7} {'WER_B':>7} {'Delta':>7}  {'95% CI':>20}  {'p-value':>10}  {'Sig':>5}")
        print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*7}  {'-'*20}  {'-'*10}  {'-'*5}")
        for a, b, label in COMPARISONS:
            if (a, ts) not in data or (b, ts) not in data:
                continue
            r = paired_bootstrap(*data[(a, ts)], *data[(b, ts)], n_bootstrap=n_bootstrap, seed=seed)
            p_str = "<1e-05" if r["p_value"] < 1e-5 else f"{r['p_value']:.5f}"
            ci_str = f"[{r['ci_95'][0]:+.2f}, {r['ci_95'][1]:+.2f}]"
            print(
                f"  {label:<38} {r['wer_a']:>6.2f}% {r['wer_b']:>6.2f}% {r['delta']:>+6.2f}%  "
                f"{ci_str:>20}  {p_str:>10}  {sig_stars(r['p_value']):>5}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance testing for ASR results"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=list(RUN_PREFIXES),
        help="Model family to test",
    )
    parser.add_argument(
        "--language", type=str, default="both", choices=["et", "sl", "both"],
        help="Language to test (default: both)",
    )
    parser.add_argument(
        "--normalization", type=str, default="both", choices=["raw", "normalized", "both"],
        help="WER variant: raw, normalized (lowercase + no punctuation), or both (default: both)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--run-seed", type=int, default=42,
        help="Training seed in the fine-tuned run names (default: 42)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=100_000,
        help="Number of bootstrap resamples (default: 100000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap random seed (default: 42)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    langs = ["et", "sl"] if args.language == "both" else [args.language]
    norms = {"raw": [False], "normalized": [True], "both": [False, True]}[args.normalization]

    for lang in langs:
        for normalize in norms:
            run_tests(
                results_dir, args.model, lang, normalize,
                args.run_seed, args.n_bootstrap, args.seed,
            )

    print(f"\n{'='*100}")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
    print("Delta = WER_B - WER_A (negative means B is better)")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
