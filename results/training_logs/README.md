# Training logs

Per-run training histories exported from Weights & Biases (projects `syntts-asr-whisper` and `syntts-asr-parakeet`) for the 16 fine-tuning runs reported in the paper. One CSV per run, named `<run name>__<W&B run id>.csv`.

- Whisper-large-v3 (Hugging Face `Seq2SeqTrainer`): `train/loss` every 25 steps, `eval/loss` every 25 optimizer steps (checkpoint selected on lowest `eval/loss`).
- Parakeet-TDT-0.6B-v3 (NeMo / PyTorch Lightning): `train_loss` every 25 steps, `val_wer` once per epoch (checkpoint selected on lowest `val_wer`, early stopping with patience 10).

`_index.json` lists the runs and their W&B states. Three Parakeet runs show the W&B state "failed"; their logs show training ran to early stopping (last epoch = best epoch + 10), so the state reflects a step after training.
