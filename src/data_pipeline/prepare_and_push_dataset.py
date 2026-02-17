"""
Prepare and push the synthetic ASR dataset to HuggingFace Hub.

Creates 8 dataset configurations (4 conditions x 2 languages), each with
train/validation/test splits. Validation and test are always CV17 dev/test.
Train varies by condition:

  cv_only_{lang}:             CV17 train only
  cv_synth_all_{lang}:        CV17 train + filtered synthetic (all 3 categories)
  cv_synth_no_morph_{lang}:   CV17 train + filtered synthetic (paraphrase + domain only)
  cv_synth_unfiltered_{lang}: CV17 train + raw synthetic (before validation)

Usage:
    uv run python -m src.data_pipeline.prepare_and_push_dataset --repo-id yuriyvnv/synthetic_asr_et_sl
    uv run python -m src.data_pipeline.prepare_and_push_dataset --repo-id yuriyvnv/synthetic_asr_et_sl --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SYNTHETIC_TEXT_DIR = DATA_DIR / "synthetic_text"
SYNTHETIC_AUDIO_DIR = DATA_DIR / "synthetic_audio"

CV17_REPO = "fixie-ai/common_voice_17_0"
LANGUAGES = {"et": "Estonian", "sl": "Slovenian"}

# Columns we keep in the final dataset
COLUMNS = ["audio", "sentence", "source", "category", "voice"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cv17_as_dataset(language_code: str, split: str) -> Dataset:
    """Load a CV17 split, keep only needed columns, normalize Audio feature."""
    logger.info(f"  Loading CV17 {split} ({language_code})...")
    ds = load_dataset(CV17_REPO, language_code, split=split)

    # Keep only audio + sentence
    ds = ds.select_columns(["audio", "sentence"])

    # Add metadata columns
    n = len(ds)
    ds = ds.add_column("source", ["cv17"] * n)
    ds = ds.add_column("category", [""] * n)
    ds = ds.add_column("voice", [""] * n)

    # Normalize Audio feature (remove fixed sampling_rate so it matches synthetic)
    ds = ds.cast_column("audio", Audio())

    logger.info(f"    {len(ds)} rows loaded")
    return ds


def load_tts_manifest(language_code: str) -> dict[str, dict]:
    """Load TTS manifest as sentence -> metadata mapping."""
    manifest_path = SYNTHETIC_AUDIO_DIR / f"tts_manifest_{language_code}.jsonl"
    manifest = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            manifest[record["sentence"]] = record
    return manifest


def build_synthetic_dataset(
    language_code: str,
    manifest: dict[str, dict],
    subset: str,
) -> Dataset:
    """Build a Dataset of synthetic audio for a given subset.

    subset: 'all' | 'no_morph' | 'unfiltered'
    """
    if subset in ("all", "no_morph"):
        # Finalized sentences from Phase 4
        source_path = SYNTHETIC_TEXT_DIR / f"synthetic_text_{language_code}.jsonl"
        records = []
        with open(source_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if subset == "no_morph" and r.get("category") == "morphological":
                    continue
                records.append(r)

    elif subset == "unfiltered":
        # Raw unique sentences from Phase 1 (deduplicated)
        seen = set()
        records = []
        for cat in ["paraphrase", "domain", "morphological"]:
            raw_path = SYNTHETIC_TEXT_DIR / f"raw_{language_code}_{cat}.jsonl"
            with open(raw_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["sentence"] not in seen:
                        seen.add(r["sentence"])
                        records.append(r)
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # Build lists for Dataset
    audio_paths = []
    sentences = []
    categories = []
    voices = []
    missing = 0

    for r in records:
        sent = r["sentence"]
        if sent not in manifest:
            missing += 1
            continue
        m = manifest[sent]
        audio_paths.append(m["audio_path"])
        sentences.append(sent)
        categories.append(r.get("category", ""))
        voices.append(m.get("voice", ""))

    if missing > 0:
        logger.warning(f"    {missing} sentences missing audio (skipped)")

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "sentence": sentences,
        "source": ["synthetic"] * len(sentences),
        "category": categories,
        "voice": voices,
    }).cast_column("audio", Audio())

    logger.info(f"    Synthetic ({subset}): {len(ds)} rows")
    return ds


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_config(config_name: str, dd: DatasetDict, expected: dict):
    """Verify a built config matches expected counts and has no contamination."""
    all_ok = True

    for split_name, expected_count in expected.items():
        actual = len(dd[split_name])
        if actual != expected_count:
            logger.error(
                f"  FAIL {config_name}/{split_name}: "
                f"expected {expected_count}, got {actual}"
            )
            all_ok = False
        else:
            logger.info(f"  OK   {config_name}/{split_name}: {actual} rows")

    # Verify no synthetic data leaked into val/test
    for split_name in ["validation", "test"]:
        sources = set(dd[split_name]["source"])
        if sources != {"cv17"}:
            logger.error(
                f"  FAIL {config_name}/{split_name}: "
                f"CONTAMINATION — sources={sources}"
            )
            all_ok = False
        else:
            logger.info(f"  OK   {config_name}/{split_name}: sources={sources}")

    # For train splits with synthetic: verify source breakdown
    if "synth" in config_name:
        train_sources = dd["train"]["source"]
        n_cv = sum(1 for s in train_sources if s == "cv17")
        n_synth = sum(1 for s in train_sources if s == "synthetic")
        logger.info(f"  OK   {config_name}/train breakdown: {n_cv} CV + {n_synth} synthetic")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and push dataset to HuggingFace"
    )
    parser.add_argument(
        "--repo-id", type=str, required=True, help="HuggingFace repo ID"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and verify without pushing"
    )
    args = parser.parse_args()

    # Expected counts for verification
    expected_synth = {
        "et": {"all": 5852, "no_morph": 3876, "unfiltered": 5996},
        "sl": {"all": 5848, "no_morph": 3863, "unfiltered": 5982},
    }
    expected_cv = {
        "et": {"train": 3157, "validation": 2653, "test": 2653},
        "sl": {"train": 1388, "validation": 1232, "test": 1242},
    }

    conditions = [
        ("cv_only", None),
        ("cv_synth_all", "all"),
        ("cv_synth_no_morph", "no_morph"),
        ("cv_synth_unfiltered", "unfiltered"),
    ]

    all_ok = True

    for lang in ["et", "sl"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {LANGUAGES[lang]} ({lang})")
        logger.info(f"{'='*60}")

        # Load CV17 splits once per language
        cv_train = load_cv17_as_dataset(lang, "train")
        cv_val = load_cv17_as_dataset(lang, "validation")
        cv_test = load_cv17_as_dataset(lang, "test")

        # Sanity check CV counts
        assert len(cv_train) == expected_cv[lang]["train"]
        assert len(cv_val) == expected_cv[lang]["validation"]
        assert len(cv_test) == expected_cv[lang]["test"]

        # Load TTS manifest once per language
        manifest = load_tts_manifest(lang)
        logger.info(f"  TTS manifest: {len(manifest)} entries")

        for condition, synth_subset in conditions:
            config_name = f"{condition}_{lang}"
            logger.info(f"\n  Building config: {config_name}")

            # Build train split
            if synth_subset is None:
                # cv_only: just CV train
                train_ds = cv_train
            else:
                # cv_synth_*: concatenate CV train + synthetic
                synth_ds = build_synthetic_dataset(lang, manifest, synth_subset)
                train_ds = concatenate_datasets([cv_train, synth_ds])

            dd = DatasetDict({
                "train": train_ds,
                "validation": cv_val,
                "test": cv_test,
            })

            # Compute expected train count
            if synth_subset is None:
                exp_train = expected_cv[lang]["train"]
            else:
                exp_train = expected_cv[lang]["train"] + expected_synth[lang][synth_subset]

            ok = verify_config(config_name, dd, {
                "train": exp_train,
                "validation": expected_cv[lang]["validation"],
                "test": expected_cv[lang]["test"],
            })
            all_ok = all_ok and ok

            if not args.dry_run:
                if not ok:
                    logger.error(f"  SKIPPING push for {config_name} due to verification failure")
                    continue
                logger.info(f"  Pushing {config_name} to {args.repo_id}...")
                dd.push_to_hub(
                    args.repo_id,
                    config_name=config_name,
                    private=False,
                )
                logger.info(f"  Done: {config_name}")
            else:
                logger.info(f"  DRY RUN — skipping push for {config_name}")

    logger.info(f"\n{'='*60}")
    if all_ok:
        logger.info("ALL CONFIGS VERIFIED SUCCESSFULLY")
    else:
        logger.error("SOME CONFIGS FAILED VERIFICATION — check logs above")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
