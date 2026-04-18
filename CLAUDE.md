# syntts_asr — Project Context for Claude

ASR research project. Fine-tunes multilingual speech recognition models (primarily NVIDIA Parakeet-TDT-0.6B-v3, also Whisper-large-v3 and Qwen3-ASR-1.7B) with a mix of Common Voice 17 and synthetic OpenAI-TTS speech. The synthetic data is quality-filtered using the user's own WAVe multimodal embedding model.

## Environment

- **Python:** 3.12 via uv-managed interpreter. Created with `uv python install 3.12 && uv sync --python 3.12`.
- **Package manager:** `uv`. Run all Python code as `uv run python ...` — never plain `python`.
- **Venv location:** `.venv/` in repo root (managed by uv, do not edit directly).
- **GPU:** H100, ~93GB VRAM. Default to `CUDA_VISIBLE_DEVICES=0` unless told otherwise.
- **Audio standard:** 16 kHz mono, bf16-mixed precision on GPU.
- **Secrets:** `.env` holds `HF_API_KEY`, `WANDB_API_KEY`, `OPENAI_API_KEY`. Loaded via `python-dotenv`.
- **FFmpeg:** NOT installed system-wide, no sudo. `datasets>=3.0` uses `torchcodec` which `dlopen`s ffmpeg shared libs at audio-decode time. Workaround in place: canonical symlinks (`libavutil.so.60`, `libavcodec.so.62`, `libavformat.so.62`, `libavfilter.so.11`, `libavdevice.so.62`, `libswscale.so.9`, `libswresample.so.6`) point at PyAV's bundled libs in `.venv/lib/python3.12/site-packages/av.libs/`. Runs require `LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.12/site-packages/av.libs:$LD_LIBRARY_PATH` prefix. If `uv sync` rebuilds `av/`, re-run the symlinks.

## Layout

```
src/
  data_pipeline/      # Synthetic data generation (text diversification, TTS synthesis, quality filtering via WAVe)
  training/
    train_parakeet.py # Main Parakeet fine-tuning script (supports et, sl, nl, pt, pl)
    train_whisper.py  # Whisper fine-tuning
  evaluation/
    evaluate.py       # WER/CER on CV17 + FLEURS (supports zero-shot and fine-tuned)
    significance.py   # Paired bootstrap significance testing
    report_normalized.py # Normalized WER/CER comparison
scripts/
  train/                   # Per-language training entrypoints (parakeet_*.sh, whisper.sh)
  publish/                 # HF Hub upload scripts, model card updates, HF post drafts
  evaluate/                # Batch evaluation + significance testing
  data/                    # Dataset downloads + conversion
  setup.sh                 # One-time environment setup
  README.md                # How-to-run docs
results/               # Checkpoints, .nemo files, JSON eval results, wandb logs
prompts/               # Text diversification prompts for synthetic data generation
```

## Key conventions

- **Column names in training pipeline:** `audio` + `sentence`. Rename foreign columns (`text`, `ref_orig`, `transcription`) before use. `yuriyvnv/synthetic_transcript_pt` uses `text` (not `sentence`) — subsets: `cv_only`, `cv_high_quality` (48.2k, WAVe >0.5), `fully_synthetic`, `mixed_cv_synthetic` (current Parakeet-pt recipe), `mixed_cv_synthetic_all`.
- **Always cast audio to 16kHz** before `concatenate_datasets` — otherwise feature alignment fails silently.
- **Published models use full language name:** `yuriyvnv/parakeet-tdt-0.6b-dutch`, not `-nl`.
- **Experiment dumps:** full training folder (checkpoints + logs) uploaded to `yuriyvnv/experiments_parakeet`.
- **WandB project:** `syntts-asr-parakeet`.
- **Results layout:** `results/parakeet_finetune_{lang}/{config}_s{seed}/` contains `checkpoints/`, `data/`, the final `.nemo`, and `test_results.json`.

## Fine-tuning conventions (Parakeet)

- Optimizer: AdamW (β=0.9/0.98, wd=0.001)
- LR: 5e-5 peak, CosineAnnealing, 10% warmup, min 1e-6
- Batch size: 64 (short audio) or 32 (long audio, >20s samples)
- Gradient clipping: 1.0
- Precision: bf16-mixed
- Early stopping: patience 10 on `val_wer`
- Save top-3 checkpoints + `last.ckpt`

## Known pinned-version quirks

- **NeMo (currently 2.7.2):** Was 2.2.1 until `qwen-asr` forced `transformers==4.57.6`, which required NeMo ≥ 2.3.0. The CUDA-graph-decoder + bf16 workaround at `train_parakeet.py:310-314` is now redundant (fixed in NeMo 2.3.0, PR #12938) but harmless — leave it until verified unnecessary.
- **Dep cascade triggered by `qwen-asr`:** `qwen-asr==0.0.6` pins `transformers==4.57.6` → forces NeMo ≥ 2.3 → forces `fsspec==2024.12.0` → forces `datasets>=3.0.0` → forces `pyarrow>=15`. The `datasets<3.0.0` and `pyarrow<15` pins in `pyproject.toml` have been removed. Do not re-add them.
- **`datasets>=3.0` torchcodec/ffmpeg:** audio decode is via torchcodec, which dlopens system ffmpeg libs. See Environment section for the PyAV symlink workaround — required for every run that reads HF audio datasets.
- **FLEURS loading:** Try `trust_remote_code=True` first; fall back to `revision="refs/convert/parquet"` if that fails.
- **BIGOS v2 (`amu-cai/pl-asr-bigos-v2`):** Requires `trust_remote_code=True`. Configs are per-source or `"all"`. Contains samples up to 61s long (PolyAI Minds14) — consider batch size 32 to avoid RNNT OOM.
- **texterrors 0.5.1:** Needs Python.h to compile. Use uv-managed Python, not system Python 3.11 (which lacks dev headers on this cluster and sudo isn't available).

## Published models (as of last session)

| Model | Val WER | Test WER |
|---|---|---|
| `yuriyvnv/parakeet-tdt-0.6b-dutch` | 3.73% | 5.33% |
| `yuriyvnv/parakeet-tdt-0.6b-portuguese` | 9.62% | 10.71% |
| `yuriyvnv/parakeet-tdt-0.6b-estonian` | — | 21.03% |
| `yuriyvnv/parakeet-tdt-0.6b-slovenian` | — | 11.56% |
| `yuriyvnv/parakeet-tdt-0.6b-polish` | 6.07% | 11.81% (worse than zero-shot 9.72%, do not publicize) |

## User preferences

- Prefers terse, direct answers — no preamble, no trailing summaries.
- Runs long training jobs in their own shell, not via the assistant's Bash tool. Give the full command and let them run it.
- Wants to verify behavior before committing (e.g., "are you sure?" checks). When uncertain, run a small verification rather than claiming correctness.
- Dislikes destructive operations without confirmation (don't `rm -rf` without explicit approval).
- Uses HF posts for visibility — plain text only, no markdown, flag emojis for scanning.
