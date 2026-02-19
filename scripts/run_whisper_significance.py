"""
Run paired bootstrap significance tests for Whisper models.

Compares cv_only vs cv_synth_all and cv_only vs cv_synth_no_morph
for Slovenian on CV17 test and FLEURS test.

Usage:
    uv run python scripts/run_whisper_significance.py
"""

import json
from pathlib import Path

import numpy as np
from jiwer import process_words

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_BOOTSTRAP = 100_000
SEED = 42


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_word_errors(data):
    """Extract per-sentence word error counts and reference word counts."""
    errors = []
    ref_counts = []
    for s in data["per_sentence"]:
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


def paired_bootstrap(errors_a, ref_a, errors_b, ref_b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    assert n == len(errors_b)

    wer_a = errors_a.sum() / ref_a.sum() * 100
    wer_b = errors_b.sum() / ref_b.sum() * 100
    observed_delta = wer_b - wer_a

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_a = errors_a[idx].sum() / ref_a[idx].sum() * 100
        boot_b = errors_b[idx].sum() / ref_b[idx].sum() * 100
        deltas.append(boot_b - boot_a)

    deltas = np.array(deltas)
    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    if observed_delta < 0:
        p = np.mean(deltas >= 0)
    elif observed_delta > 0:
        p = np.mean(deltas <= 0)
    else:
        p = 1.0

    return {
        "wer_a": round(wer_a, 2),
        "wer_b": round(wer_b, 2),
        "delta": round(observed_delta, 2),
        "ci": (round(ci_lo, 2), round(ci_hi, 2)),
        "p": p,
    }


def sig(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def find_result(model_key, lang, test_set):
    filename = f"{model_key}_{lang}_{test_set}.json"
    matches = list(RESULTS_DIR.glob(f"**/{filename}"))
    if not matches:
        return None
    return load_json(matches[0])


# ---- Models to compare ----
WHISPER_MODELS = {
    "zero-shot": "whisper-large-v3",
    "cv_only": "whisper-large-v3-cv_only_sl-seed42",
    "cv_synth_no_morph": "whisper-large-v3-cv_synth_no_morph_sl-seed42",
    "cv_synth_all": "whisper-large-v3-cv_synth_all_sl-seed42",
}

COMPARISONS = [
    ("zero-shot", "cv_only", "Zero-shot vs CV-only"),
    ("zero-shot", "cv_synth_no_morph", "Zero-shot vs CV+Synth No Morph"),
    ("zero-shot", "cv_synth_all", "Zero-shot vs CV+Synth All"),
    ("cv_only", "cv_synth_no_morph", "CV-only vs CV+Synth No Morph"),
    ("cv_only", "cv_synth_all", "CV-only vs CV+Synth All"),
    ("cv_synth_no_morph", "cv_synth_all", "CV+Synth No Morph vs CV+Synth All"),
]

TEST_SETS = ["cv17_test", "fleurs_test"]

# ---- Run ----
print(f"\nPaired Bootstrap Significance — Whisper-large-v3 Slovenian (n={N_BOOTSTRAP})")
print("=" * 95)

# Preload
data = {}
for name, key in WHISPER_MODELS.items():
    for ts in TEST_SETS:
        d = find_result(key, "sl", ts)
        if d:
            errors, refs = get_word_errors(d)
            data[(name, ts)] = (errors, refs)
            print(f"  Loaded: {name} / {ts} ({len(errors)} sentences, WER={d['wer']}%)")
        else:
            print(f"  MISSING: {name} / {ts}")

print()

for ts in TEST_SETS:
    print(f"\n--- {ts} ---")
    print(f"  {'Comparison':<38} {'WER_A':>7} {'WER_B':>7} {'Delta':>7}  {'95% CI':>20}  {'p-value':>10}  {'Sig':>5}")
    print(f"  {'-'*38} {'-'*7} {'-'*7} {'-'*7}  {'-'*20}  {'-'*10}  {'-'*5}")

    for a, b, label in COMPARISONS:
        if (a, ts) not in data or (b, ts) not in data:
            print(f"  {label:<38} {'(missing data)':>50}")
            continue

        ea, ra = data[(a, ts)]
        eb, rb = data[(b, ts)]

        result = paired_bootstrap(ea, ra, eb, rb)

        p_str = f"<1e-05" if result["p"] < 1e-5 else f"{result['p']:.5f}"
        ci_str = f"[{result['ci'][0]:+.2f}, {result['ci'][1]:+.2f}]"

        print(
            f"  {label:<38} {result['wer_a']:>6.2f}% {result['wer_b']:>6.2f}% "
            f"{result['delta']:>+6.2f}%  {ci_str:>20}  {p_str:>10}  {sig(result['p']):>5}"
        )

print(f"\n{'='*95}")
print("Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
print("Delta = WER_B - WER_A (negative means B is better)")
print(f"{'='*95}\n")
