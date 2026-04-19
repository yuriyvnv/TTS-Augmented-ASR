"""
Rescore ASR evaluation JSON results with "written-form"-normalized references.

Normalization rule applied to each reference sentence:
  1. Capitalize first character if it's lowercase (skip if already uppercase,
     non-letter, or empty).
  2. If the last non-whitespace character is not already a terminal symbol
     (.!?…) or a closing bracket/quote (")]}'"»"”), append a period.

Hypotheses are NOT modified — the metric measures how close the raw model
output is to the intended written form of the reference.

Usage:
    uv run python -m src.evaluation.score_written_form
    uv run python -m src.evaluation.score_written_form --results-dir ./results
    uv run python -m src.evaluation.score_written_form --also-normalize-hyps
    uv run python -m src.evaluation.score_written_form --write-alongside

Input: every `*.json` under `--results-dir` that has a `per_sentence` array
       of `{reference, hypothesis}` objects (the schema produced by
       `src/evaluation/evaluate.py`).

Output (stdout): a before/after comparison table. With `--write-alongside`,
       also writes `<original>.written_form.json` next to each input file.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from jiwer import cer as compute_cer
from jiwer import wer as compute_wer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Characters that count as "sentence already terminated" — if a reference ends
# with any of these, we DON'T append a period.
TERMINAL_CHARS = set(".!?…")
CLOSING_CHARS = set(")]}\"'»”’›)")  # closing brackets + various close-quotes

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


# Matches any trailing run of "." or "…" characters. We then collapse to a
# single "." if the run is length>=2 OR contains the ellipsis char — so
# "foo..." / "bar…" / "baz…." / "qux.." all end up as "foo.", but a lone
# trailing "." stays untouched.
_TRAILING_DOTS_RE = re.compile(r"[.…]+$")


def normalize_written_form(sentence: str) -> str:
    """Apply the capitalize-first + single-terminal-period rule.

    Rules, in order:
      1. Strip leading/trailing whitespace.
      2. Collapse trailing "..." / "…" / ".." to a single ".".  So wordlist
         entries like "foo bar baz..." become "foo bar baz."
      3. Capitalize the first character if it's lowercase (leaves uppercase
         or non-letter starts alone).
      4. If the last non-whitespace character is NOT already a terminal
         (.!?) or closing bracket/quote (")]}'"»"”), append exactly one ".".
         If there's already a terminal, add nothing.

    Idempotent: f(f(x)) == f(x). Robust to empty/whitespace-only input.
    """
    if sentence is None:
        return ""
    s = sentence.strip()
    if not s:
        return s

    # 1. Collapse trailing run of "." / "…" to a single "." (lone "." stays "."; mixed runs collapse)
    s = _TRAILING_DOTS_RE.sub(".", s)

    # 2. Capitalize first letter
    if s[0].islower():
        s = s[0].upper() + s[1:]

    # 3. Append period if no terminal symbol / closing bracket / quote
    last = s[-1]
    if last not in TERMINAL_CHARS and last not in CLOSING_CHARS:
        s = s + "."

    return s


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------


def rescore_file(path: Path, also_normalize_hyps: bool = False) -> dict | None:
    """Read one eval JSON, recompute WER/CER with normalized references.

    Returns a row dict suitable for reporting, or None if the file lacks the
    expected schema.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_sentence = data.get("per_sentence")
    if not per_sentence:
        return None

    raw_refs = [s["reference"].strip() for s in per_sentence]
    raw_hyps = [s["hypothesis"].strip() for s in per_sentence]

    norm_refs = [normalize_written_form(r) for r in raw_refs]
    if also_normalize_hyps:
        norm_hyps = [normalize_written_form(h) for h in raw_hyps]
    else:
        norm_hyps = raw_hyps

    # Count how many references were actually changed by the rule (sanity +
    # gives a sense of how "inconsistent" the reference set is)
    n_changed = sum(1 for a, b in zip(raw_refs, norm_refs) if a != b)
    n_case_changed = sum(
        1 for a, b in zip(raw_refs, norm_refs)
        if a and b and a[:1] != b[:1] and b[:1].isupper()
    )
    n_period_added = sum(
        1 for a, b in zip(raw_refs, norm_refs)
        if len(b) == len(a) + 1 and b.endswith(".")
    )

    raw_wer = round(compute_wer(raw_refs, raw_hyps) * 100, 2)
    raw_cer = round(compute_cer(raw_refs, raw_hyps) * 100, 2)
    new_wer = round(compute_wer(norm_refs, norm_hyps) * 100, 2)
    new_cer = round(compute_cer(norm_refs, norm_hyps) * 100, 2)

    return {
        "path": str(path.relative_to(path.parent.parent)),
        "n_samples": len(raw_refs),
        "n_refs_changed": n_changed,
        "n_case_changes": n_case_changed,
        "n_periods_added": n_period_added,
        "raw_wer": raw_wer,
        "raw_cer": raw_cer,
        "wf_wer": new_wer,
        "wf_cer": new_cer,
        "delta_wer": round(new_wer - raw_wer, 2),
        "delta_cer": round(new_cer - raw_cer, 2),
    }


def write_alongside(path: Path, also_normalize_hyps: bool) -> Path:
    """Write a sibling JSON with normalized refs + (optionally) hyps and
    the recomputed per-sentence / aggregate metrics."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = dict(data)  # shallow copy

    per_sentence = data.get("per_sentence", [])
    new_per_sentence = []
    for s in per_sentence:
        nr = normalize_written_form(s.get("reference", ""))
        nh = normalize_written_form(s.get("hypothesis", "")) if also_normalize_hyps else s.get("hypothesis", "")
        # Per-sample WER on normalized pair
        from jiwer import wer as _wer
        if not nr.strip():
            sent_wer = 0.0 if not nh.strip() else 100.0
        else:
            sent_wer = round(_wer(nr, nh) * 100, 2)
        new_per_sentence.append({
            "reference": nr,
            "hypothesis": nh,
            "wer": sent_wer,
        })

    refs = [e["reference"] for e in new_per_sentence]
    hyps = [e["hypothesis"] for e in new_per_sentence]
    out["wer"] = round(compute_wer(refs, hyps) * 100, 2)
    out["cer"] = round(compute_cer(refs, hyps) * 100, 2)
    out["per_sentence"] = new_per_sentence
    out["normalization"] = {
        "rule": "written_form: capitalize-first + add-terminal-period if absent",
        "also_normalize_hyps": also_normalize_hyps,
    }

    out_path = path.with_suffix(".written_form.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Rescore evaluation JSONs with written-form-normalized references."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory to search (recursively) for *.json with per_sentence arrays.",
    )
    parser.add_argument(
        "--also-normalize-hyps", action="store_true",
        help="Also apply the same normalization to hypotheses (symmetric scoring).",
    )
    parser.add_argument(
        "--write-alongside", action="store_true",
        help="Write a sibling <file>.written_form.json with the new per-sentence "
             "and aggregate metrics.",
    )
    args = parser.parse_args()

    logger.info(f"Scanning {args.results_dir} for eval JSONs...")
    rows: list[dict] = []
    for path in sorted(args.results_dir.rglob("*.json")):
        # Skip sibling outputs we write ourselves
        if path.name.endswith(".written_form.json"):
            continue
        try:
            row = rescore_file(path, also_normalize_hyps=args.also_normalize_hyps)
        except Exception as e:
            logger.warning(f"  skip {path}: {type(e).__name__}: {e}")
            continue
        if row is None:
            continue
        rows.append(row)
        if args.write_alongside:
            out = write_alongside(path, also_normalize_hyps=args.also_normalize_hyps)
            logger.info(f"  wrote {out.relative_to(args.results_dir)}")

    if not rows:
        logger.error("No scoreable JSONs found.")
        return

    # Sort for readable output
    rows.sort(key=lambda r: r["path"])

    print()
    print("=" * 120)
    print("  Written-form rescoring "
          f"({'refs+hyps' if args.also_normalize_hyps else 'refs only'} normalized)")
    print("  Rule: first letter UPPER if lowercase; append '.' if last char not in .!?…")
    print("        or closing-bracket/quote. Hypotheses unchanged.")
    print("=" * 120)
    print(
        f"  {'File':<70}  {'n':>5}  {'refs Δ':>6}  "
        f"{'raw WER':>8} {'wf WER':>8} {'ΔWER':>7}  {'raw CER':>8} {'wf CER':>8} {'ΔCER':>7}"
    )
    print(f"  {'-'*70}  {'-'*5}  {'-'*6}  {'-'*8} {'-'*8} {'-'*7}  {'-'*8} {'-'*7} {'-'*7}")
    for r in rows:
        print(
            f"  {r['path']:<70}  {r['n_samples']:>5}  {r['n_refs_changed']:>6}  "
            f"{r['raw_wer']:>7.2f}% {r['wf_wer']:>7.2f}% {r['delta_wer']:>+6.2f}  "
            f"{r['raw_cer']:>7.2f}% {r['wf_cer']:>7.2f}% {r['delta_cer']:>+6.2f}"
        )
    print("=" * 120)
    print(f"  Negative Δ = normalization IMPROVED the score (i.e. model was already in written form)")
    print(f"  Positive Δ = normalization WORSENED the score (i.e. raw refs were closer to the model output)")
    print()


if __name__ == "__main__":
    main()
