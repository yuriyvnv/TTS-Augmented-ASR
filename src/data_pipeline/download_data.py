"""
Download CommonVoice 17 and FLEURS datasets, extract seed and test texts.

Outputs:
    data/seeds_{lang}.txt       — CV17 train sentences (seeds for text generation)
    data/test_texts_{lang}.txt  — CV17 test + dev + FLEURS test sentences (leakage prevention)
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Dataset configs
CV17_REPO = "fixie-ai/common_voice_17_0"
FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "refs/convert/parquet"

FLEURS_LANG = {"et": "et_ee", "sl": "sl_si"}


def _load_fleurs_test(language_code: str) -> list[str]:
    """Download FLEURS test parquet files directly and extract transcriptions."""
    fleurs_lang = FLEURS_LANG[language_code]
    sentences = []
    for i in range(10):  # parquet shards 0000, 0001, ...
        filename = f"{fleurs_lang}/test/{i:04d}.parquet"
        try:
            path = hf_hub_download(
                repo_id=FLEURS_REPO,
                filename=filename,
                repo_type="dataset",
                revision=FLEURS_REVISION,
            )
            df = pd.read_parquet(path)
            if "raw_transcription" in df.columns:
                sentences.extend(df["raw_transcription"].dropna().str.strip().tolist())
            elif "transcription" in df.columns:
                sentences.extend(df["transcription"].dropna().str.strip().tolist())
        except Exception:
            break  # no more shards
    return [s for s in sentences if s]


def download_and_extract(language_code: str, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    # --- CV17: train sentences → seeds ---
    print(f"Downloading CV17 train ({language_code})...")
    cv_train = load_dataset(CV17_REPO, language_code, split="train")
    seeds = [row["sentence"] for row in cv_train if row["sentence"].strip()]

    seeds_path = os.path.join(output_dir, f"seeds_{language_code}.txt")
    with open(seeds_path, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(s.strip() + "\n")
    print(f"  Seeds: {len(seeds)} sentences → {seeds_path}")

    # --- CV17 test + dev + FLEURS test → test texts for leakage prevention ---
    test_sentences = set()

    print(f"Downloading CV17 test ({language_code})...")
    cv_test = load_dataset(CV17_REPO, language_code, split="test")
    for row in cv_test:
        if row["sentence"].strip():
            test_sentences.add(row["sentence"].strip())

    print(f"Downloading CV17 validation ({language_code})...")
    cv_dev = load_dataset(CV17_REPO, language_code, split="validation")
    for row in cv_dev:
        if row["sentence"].strip():
            test_sentences.add(row["sentence"].strip())

    print(f"Downloading FLEURS test ({FLEURS_LANG[language_code]})...")
    fleurs_sentences = _load_fleurs_test(language_code)
    for s in fleurs_sentences:
        test_sentences.add(s)
    print(f"  FLEURS test: {len(fleurs_sentences)} sentences")

    test_path = os.path.join(output_dir, f"test_texts_{language_code}.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        for s in sorted(test_sentences):
            f.write(s + "\n")
    print(f"  Test texts: {len(test_sentences)} sentences → {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets and extract texts")
    parser.add_argument("--language", type=str, required=True, choices=["et", "sl", "both"])
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    langs = ["et", "sl"] if args.language == "both" else [args.language]
    for lang in langs:
        download_and_extract(lang, args.output_dir)


if __name__ == "__main__":
    main()
