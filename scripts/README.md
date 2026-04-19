# `scripts/` — How to Run Each Workflow

All scripts must be run from the **repository root** (not from inside `scripts/`), so that `uv run` finds `pyproject.toml` and Python module paths resolve correctly.

## Directory map

| Folder | Purpose |
|---|---|
| [`train/`](train/) | Training entrypoints (one bash script per language) |
| [`publish/`](publish/) | HF Hub upload + model cards + announcements |
| [`evaluate/`](evaluate/) | Batch evaluation + significance testing |
| [`data/`](data/) | Data downloads + format conversion |
| [`docker/`](docker/) | Containerised workflow for Qwen3-ASR fine-tuning (survives SSH disconnects). See [`docker/README.md`](docker/README.md). |
| [`setup.sh`](setup.sh) | One-time environment setup (native, non-Docker) |

---

## 1. One-time setup

```bash
bash scripts/setup.sh
```

Installs `uv`, creates the `.venv/`, runs `uv sync --python 3.12`, and applies the PyAV symlink workaround for ffmpeg. Only run once per machine.

Then log in to HuggingFace and WandB:

```bash
uv run huggingface-cli login
uv run wandb login
```

---

## 2. Fine-tuning

Scripts in [`train/`](train/) are self-contained: they set hyperparameters, GPU index, and output directory as variables at the top, then call `uv run python -m src.training.train_parakeet` (or `train_whisper`).

### Paper languages (Estonian, Slovenian)

```bash
bash scripts/train/parakeet.sh   # edit LANGUAGE / CONFIG / SEED at the top
bash scripts/train/whisper.sh
```

Valid Parakeet configs for paper experiments:
- `cv_only_et`, `cv_only_sl`
- `cv_synth_all_et`, `cv_synth_all_sl`
- `cv_synth_no_morph_et`, `cv_synth_no_morph_sl`
- `cv_synth_unfiltered_et`, `cv_synth_unfiltered_sl`

### Add-on languages

```bash
bash scripts/train/parakeet_nl.sh    # Dutch: CV17 + synthetic_transcript_nl
bash scripts/train/parakeet_pt.sh    # Portuguese: mixed_cv_synthetic config
```

### Qwen3-ASR-1.7B fine-tunes (Docker only)

These run inside the Docker container ([`docker/README.md`](docker/README.md)) because they need flash-attn + ffmpeg + matched CUDA, none of which are available natively on this host.

```bash
bash scripts/docker/up.sh                   # start container (one-time per session)
bash scripts/docker/train.sh pt             # → scripts/train/qwen_pt.sh
bash scripts/docker/train.sh nl             # → scripts/train/qwen_nl.sh
```

Each `qwen_*.sh` calls `python -m src.training.train_qwen3_asr` with hyperparameters fixed at the top of the script. Both apply written-form normalisation (`normalize_written_form`) to train / val / test references and auto-push the best checkpoint to `yuriyvnv/Qwen3-ASR-1.7B-{LANG}`.

For Polish there is no shell script (the add-on run used BIGOS v2 filtered; the Polish model underperformed zero-shot, so it's not published). To reproduce:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.training.train_parakeet \
    --language pl --config bigos_cased_pl \
    --output-dir ./results/parakeet_finetune_pl/bigos_cased_pl_s42 \
    --seed 42 --batch-size 32 --learning-rate 5e-5 --warmup-ratio 0.10 \
    --max-epochs 100 --early-stopping-patience 10 \
    --push-to-hub --hub-repo-id yuriyvnv/experiments_parakeet
```

### Training outputs

Each run creates:
```
results/parakeet_finetune_{lang}/{config}_s{seed}/
├── checkpoints/           # top-3 + last.ckpt
├── data/                  # NeMo manifests + converted 16kHz WAVs
├── wandb/                 # WandB local logs
├── *.nemo                 # final model (saved from best checkpoint)
└── test_results.json      # CV17 test WER/CER, auto-run after training
```

---

## 3. Evaluation

### Single model, single language

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.evaluation.evaluate \
    --model parakeet-tdt-0.6b-v3 \
    --model-path ./results/parakeet_finetune_nl/cv_synth_nl_s42/parakeet-tdt-cv_synth_nl-seed42.nemo \
    --language nl \
    --test-sets cv17_test cv17_validation fleurs_test \
    --batch-size 32
```

Omit `--model-path` for zero-shot evaluation.

Use `--language all` to evaluate on every supported language in one run.

### Batch-evaluate all Whisper experiments

```bash
uv run python scripts/evaluate/whisper_all.py
```

### Statistical significance

Paired bootstrap between zero-shot and fine-tuned conditions:

```bash
uv run python scripts/evaluate/whisper_significance.py              # raw text
uv run python scripts/evaluate/whisper_significance_normalized.py   # jiwer-normalized
```

### Qwen3-ASR zero-shot baseline

Re-measures the un-fine-tuned base model on CV17 + CV22 test sets with the SAME code path the fine-tuned eval uses (`evaluate_model` from `train_qwen3_asr.py`, normalised refs, greedy decode). Required for an apples-to-apples comparison in the model card. Output JSONs land in `results/qwenV3/<lang>/qwen3-asr-1.7b_<lang>_<set>_baseline.json` and are picked up automatically by the publish script.

```bash
docker exec qwen-training bash -c \
    "cd /workspace && uv run python scripts/evaluate/qwen_pt_zero_shot_baseline.py"
```

---

## 4. Publishing to HuggingFace Hub

### Upload a fine-tuned Parakeet model

One script per language. Each uploads the `.nemo` file plus a production-ready model card with benchmark results.

```bash
uv run python scripts/publish/parakeet_nl.py
uv run python scripts/publish/parakeet_pt.py
uv run python scripts/publish/parakeet_pl.py
```

### Update model card READMEs without re-uploading the weights

```bash
uv run python scripts/publish/update_readmes.py
```

### Upload / refresh the Qwen3-ASR model card

`qwen_pt.py` reads `results/qwen3_finetune_pt/<run>/test_results.json` plus any `results/qwenV3/pt/*_baseline.json` files and renders a friendly model card (badges, comparison table, normalisation explainer, acknowledgements). Used after the in-training auto-push if you want to refresh the README only.

```bash
uv run python scripts/publish/qwen_pt.py --dry-run        # preview rendered README
uv run python scripts/publish/qwen_pt.py --readme-only    # push README, no weights
uv run python scripts/publish/qwen_pt.py                  # push README + weights
```

### Draft HuggingFace announcement post

The markdown draft for the multi-language launch post is at [`publish/hf_post_parakeet.md`](publish/hf_post_parakeet.md). HF posts do not render markdown — copy the flag-emoji plain-text version when posting.

### Create the `experiments_parakeet` HF repo (once)

```bash
uv run python scripts/publish/create_experiments_repo.py
```

---

## 5. Data prep

Download CV17 + synthetic datasets and convert to NeMo-compatible format:

```bash
uv run python scripts/data/download_and_convert.py
```

Download cached Whisper baseline weights:

```bash
uv run python scripts/data/download_whisper_models.py
```

---

## Conventions

- **Always use `uv run`** — never plain `python`, or the wrong interpreter will be picked up.
- **GPU index**: default to `CUDA_VISIBLE_DEVICES=0`. Change only if another user is on GPU 0.
- **Secrets**: `HF_API_KEY`, `WANDB_API_KEY`, `OPENAI_API_KEY` in `.env`.
- **Results path**: training writes to `results/{model}_finetune_{lang}/{config}_s{seed}/`; evaluation writes to `results/{model}/{lang}/`.
- **FFmpeg workaround**: if you re-run `uv sync` and `av.libs/` is rebuilt, re-create the symlinks (see [`../CLAUDE.md`](../CLAUDE.md)).
