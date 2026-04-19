# Training

Fine-tuning scripts for three model families:

- `train_whisper.py` — Whisper-large-v3 (HF `Seq2SeqTrainer`)
- `train_parakeet.py` — NVIDIA Parakeet-TDT-0.6B-v3 (NeMo + Lightning)
- `train_qwen3_asr.py` — Alibaba Qwen3-ASR-1.7B (HF `Trainer`, custom collator + WER callback)

The Whisper / Parakeet scripts load data from `yuriyvnv/synthetic_asr_et_sl` (paper experiments). The Qwen3-ASR script loads from CV22 / `synthetic_transcript_pt` / `synthetic_transcript_nl` and is intended to run inside the Docker container ([`scripts/docker/`](../../scripts/docker/)).

## Whisper-large-v3

Full fine-tuning via HuggingFace Transformers `Seq2SeqTrainer`.

```bash
uv run python -m src.training.train_whisper \
    --language et --config cv_synth_all_et \
    --output-dir ./results/whisper_et_cv_synth_all_s42 --seed 42
```

**Key defaults** (H100 80GB):
- Batch size: 32
- LR: 1e-5, warmup: 500 steps
- Max steps: 4000, eval every 500 steps
- bf16, gradient checkpointing, SDPA attention
- Best checkpoint selected by eval_loss (WER evaluated separately after training)

## Parakeet-TDT-0.6B-v3

Full fine-tuning via NeMo + PyTorch Lightning.

### Shell script (recommended)

Edit the settings at the top of `scripts/train_parakeet.sh` to set language, config, batch size, LR, etc., then run:

```bash
# Prerequisites
huggingface-cli login
wandb login

# Edit scripts/train_parakeet.sh to set LANGUAGE, CONFIG, etc.
./scripts/train_parakeet.sh
```

### Direct command

```bash
uv run python -m src.training.train_parakeet \
    --language et --config cv_synth_all_et \
    --output-dir ./results/parakeet_finetune_et/cv_synth_all_et_s42 \
    --seed 42 --push-to-hub
```

**Key defaults** (H100 80GB):
- Batch size: 32
- LR: 5e-5 peak, warmup: 10% of total steps, cosine annealing to min_lr 1e-6
- Max epochs: 100, early stopping (patience: 10 on val_wer)
- bf16-mixed, gradient clipping 1.0
- Saves top-3 checkpoints by `val_wer`
- Pushes final model to `yuriyvnv/parakeet-tdt-0.6b-v3-{language}`

**Note**: First run converts HF dataset to NeMo JSONL manifests + WAV files (stored in `--data-dir` or `output-dir/data/`). Subsequent runs reuse existing manifests.

## Dataset Configurations

| Config | Train contents |
|--------|---------------|
| `cv_only_{lang}` | CV17 train only |
| `cv_synth_all_{lang}` | CV17 train + synthetic (all 3 categories) |
| `cv_synth_no_morph_{lang}` | CV17 train + synthetic (paraphrase + domain only) |
| `cv_synth_unfiltered_{lang}` | CV17 train + raw synthetic (before validation) |

## Qwen3-ASR-1.7B

Full fine-tune (no LoRA), bf16, follows the [official QwenLM SFT recipe](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) with our local hyperparameters. Run only inside the Docker container — flash-attn 2 and matched CUDA 12.8 nvcc are required and not present on the host.

```bash
bash scripts/docker/up.sh
bash scripts/docker/train.sh pt        # → scripts/train/qwen_pt.sh
bash scripts/docker/train.sh nl        # → scripts/train/qwen_nl.sh
```

**Per-language datasets:**

| `--language` | `--dataset` | Train source |
|---|---|---|
| `pt` | `cv22` | `fsicoli/common_voice_22_0` (pt) |
| `pt` | `synthetic_pt_high_quality` | `yuriyvnv/synthetic_transcript_pt` (subset `cv_high_quality`, WAVe-filtered CV17-pt) |
| `nl` | `cv22` | `fsicoli/common_voice_22_0` (nl) |
| `nl` | `mixed_nl` | `yuriyvnv/synthetic_transcript_nl` (all ~34.9k synthetic) **+** CV22-nl train, concatenated and shuffled |

**Training-time invariants:**

- Targets formatted as `language {Name}<asr_text>{transcript}<|im_end|>` (matches official SFT prompt). Prefix masked to -100 so loss flows only on the transcript.
- Right-padding forced per-call in `DataCollatorQwen3ASR` (Qwen3ASRProcessor's default left-padding shifts the prefix-mask offset — see [QwenLM/Qwen3-ASR#70](https://github.com/QwenLM/Qwen3-ASR/issues/70)).
- All references go through `normalize_written_form` (capitalise first letter, collapse trailing dots, append terminal period).
- `MAX_SENTENCE_CHARS=500` filter rejects pathological transcripts that cause 50+ GB single-tensor OOMs at lm_head ([QwenLM/Qwen3-ASR#91](https://github.com/QwenLM/Qwen3-ASR/issues/91)).
- `WERCallback` runs greedy generation on the full validation set at the end of each epoch and logs `eval/wer`, `eval/cer` to wandb.
- Best checkpoint (by `eval_loss`) auto-pushes to `yuriyvnv/Qwen3-ASR-1.7B-{LANG}` in two stages: weights + minimal README first, then the full README with the apples-to-apples zero-shot vs fine-tuned comparison once eval finishes.
