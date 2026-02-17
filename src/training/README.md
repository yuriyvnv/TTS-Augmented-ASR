# Training

Fine-tuning scripts for Whisper-large-v3 and Parakeet-TDT-0.6B-v3 on the synthetic ASR dataset.

Both scripts load data from `yuriyvnv/synthetic_asr_et_sl` on HuggingFace.

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
- LR: 1e-5, warmup: 200 steps, cosine annealing (min_lr: 1e-6)
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
