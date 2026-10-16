# Training

Full fine-tuning of Whisper-large-v3 and Parakeet-TDT-0.6B-v3. Both scripts load their data from the Hub dataset [`yuriyvnv/synthetic_asr_et_sl`](https://huggingface.co/datasets/yuriyvnv/synthetic_asr_et_sl) (text column: `sentence`).

## Launchers (paper settings)

```bash
./scripts/train_whisper.sh                       # all 8 language/config pairs
./scripts/train_parakeet.sh sl cv_only_sl        # one pair
HUB_REPO=<your-user>/<repo> ./scripts/train_whisper.sh et cv_synth_all_et   # also upload the run folder
```

Outputs:
- Whisper: `results/whisper_finetune_{lang}/whisper-large-v3-{config}-seed42/` (best model at the folder root)
- Parakeet: `results/parakeet_finetune_{lang}/{config}_s42/parakeet-tdt-{config}-seed42.nemo`

## Whisper-large-v3

Hugging Face Transformers `Seq2SeqTrainer`.

```bash
uv run python -m src.training.train_whisper \
    --language et --config cv_synth_all_et \
    --output-dir ./results/whisper_finetune_et/whisper-large-v3-cv_synth_all_et-seed42 --seed 42
```

Defaults (match the paper):
- Batch 64 x 2 gradient accumulation = 128 effective, eval batch 8
- AdamW (fused), LR 5e-5, 10% warmup, linear decay, 5 epochs
- Eval and save every 25 steps; best checkpoint by `eval_loss` (WER is computed separately with `src.evaluation.evaluate`)
- bf16, gradient checkpointing, SDPA attention

## Parakeet-TDT-0.6B-v3

NeMo + PyTorch Lightning.

```bash
uv run python -m src.training.train_parakeet \
    --language et --config cv_synth_all_et \
    --output-dir ./results/parakeet_finetune_et/cv_synth_all_et_s42 --seed 42
```

Defaults (match the paper):
- Batch 32
- AdamW (betas 0.9/0.98, weight decay 1e-3), LR 5e-5 peak, 10% warmup, cosine annealing to 1e-6
- Up to 100 epochs, early stopping with patience 10 on `val_wer`
- bf16-mixed, gradient clipping 1.0, top-3 checkpoints by `val_wer`; the best one is exported to `.nemo`

The first run converts the HF dataset to NeMo JSONL manifests + WAV files under `--data-dir` (default `output-dir/data/`). Later runs reuse them.

NeMo 2.2.1 note: CUDA-graph greedy decoding is disabled because, combined with bf16-mixed, it corrupts decoder state during validation (fixed upstream in NeMo 2.3.0).

## Optional upload

Both scripts take `--push-to-hub --hub-repo-id <user>/<repo>` to upload the run folder after training.

## Dataset configurations

| Config | Train contents |
|--------|---------------|
| `cv_only_{lang}` | CV17 train only |
| `cv_synth_all_{lang}` | CV17 train + validated synthetic (all 3 categories) |
| `cv_synth_no_morph_{lang}` | CV17 train + validated synthetic (paraphrase + domain only) |
| `cv_synth_unfiltered_{lang}` | CV17 train + raw synthetic (before validation) |
