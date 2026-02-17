# Data Pipeline

Scripts for generating, validating, and publishing the synthetic ASR dataset.
Run in order: download → text diversification → TTS synthesis → push to Hub.

## Scripts

### 1. `download_data.py` — Download source datasets

Downloads CommonVoice 17 and FLEURS from HuggingFace. Extracts seed sentences (CV17 train) for text generation and test sentences (CV17 test + dev + FLEURS test) for leakage prevention.

```bash
uv run python -m src.data_pipeline.download_data --language et
uv run python -m src.data_pipeline.download_data --language sl
```

**Outputs:**
- `data/seeds_{lang}.txt` — CV17 train sentences (used as seeds for paraphrase generation)
- `data/test_texts_{lang}.txt` — CV17 test + dev + FLEURS test sentences (used to prevent data leakage)

### 2. `text_diversification.py` — LLM text generation + validation

4-phase pipeline that generates diverse synthetic sentences using GPT-5-mini:

| Phase | Description | Output |
|-------|-------------|--------|
| 1. Generate | Async LLM generation across 3 categories | `data/synthetic_text/raw_{lang}_{category}.jsonl` |
| 2. Validate | LLM-as-judge quality assessment | `data/synthetic_text/validated_{lang}.jsonl` |
| 3. Regenerate | Corrective re-generation of failures | `data/synthetic_text/regenerated_{lang}.jsonl` |
| 4. Finalize | Dedup + test-set leakage removal | `data/synthetic_text/synthetic_text_{lang}.jsonl` |

**3 categories:**
- **Paraphrase** (~2,000/lang): Rewrites of CV17 sentences for lexical/syntactic diversity
- **Domain Expansion** (~2,000/lang): Sentences across 10 underrepresented domains
- **Morphological Diversity** (~2,000/lang): Sentences targeting rare case/number forms

```bash
uv run python -m src.data_pipeline.text_diversification --language et --phase all
uv run python -m src.data_pipeline.text_diversification --language sl --phase all
```

Each phase is independently resumable. Run individual phases with `--phase generate|validate|regenerate|finalize`.

### 3. `tts_synthesis.py` — OpenAI TTS synthesis

Synthesizes all unique sentences using `gpt-4o-mini-tts` with 11 voices cycled deterministically (sha256 hash modulo). Outputs 24kHz WAV files (resampling handled downstream by training frameworks).

```bash
uv run python -m src.data_pipeline.tts_synthesis --language et
uv run python -m src.data_pipeline.tts_synthesis --language sl
uv run python -m src.data_pipeline.tts_synthesis --language both
uv run python -m src.data_pipeline.tts_synthesis --language et --concurrency 50
```

**Outputs:**
- `data/synthetic_audio/{lang}/*.wav` — Audio files (named by sentence hash)
- `data/synthetic_audio/tts_manifest_{lang}.jsonl` — Sentence-to-audio mapping with metadata
- `data/synthetic_audio/tts_stats_{lang}.json` — Summary statistics

Resumable: skips sentences whose audio already exists on disk.

### 4. `prepare_and_push_dataset.py` — Build and push HuggingFace dataset

Creates 8 dataset configurations (4 conditions x 2 languages) and pushes to HuggingFace Hub. Each config has train/validation/test splits. Validation and test are always CV17 dev/test (no synthetic data).

| Config | Train contents |
|--------|---------------|
| `cv_only_{lang}` | CV17 train only |
| `cv_synth_all_{lang}` | CV17 train + filtered synthetic (all 3 categories) |
| `cv_synth_no_morph_{lang}` | CV17 train + filtered synthetic (paraphrase + domain only) |
| `cv_synth_unfiltered_{lang}` | CV17 train + raw synthetic (before validation) |

```bash
# Dry run (build + verify without pushing)
uv run python -m src.data_pipeline.prepare_and_push_dataset --repo-id yuriyvnv/synthetic_asr_et_sl --dry-run

# Push to Hub
uv run python -m src.data_pipeline.prepare_and_push_dataset --repo-id yuriyvnv/synthetic_asr_et_sl
```

Verification checks: exact row counts, no synthetic contamination in val/test, source breakdown in train.

## Data Flow

```
CV17 train sentences ──→ text_diversification.py ──→ tts_synthesis.py ──→ prepare_and_push_dataset.py
    (seeds)                  (synthetic text)          (synthetic audio)       (HuggingFace Hub)
```

## Generated Data Summary

| | Estonian (et) | Slovenian (sl) |
|---|---|---|
| Raw sentences | 6,001 | 6,001 |
| Unique raw | 5,996 | 5,982 |
| Finalized (post-validation) | 5,852 | 5,848 |
| Finalized (no morph) | 3,876 | 3,863 |
| TTS audio files | 7,743 | 7,365 |
| Total audio duration | 14.14 hrs | 13.51 hrs |
| TTS model | gpt-4o-mini-tts | gpt-4o-mini-tts |
| Voices | 11 (alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse) |
