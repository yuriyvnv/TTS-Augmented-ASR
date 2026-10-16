# Evaluation

Evaluate Whisper-large-v3 and Parakeet-TDT-0.6B-v3 on CommonVoice 17 and FLEURS for Estonian and Slovenian, then compare systems with paired bootstrap tests.

## `evaluate.py`

### Zero-shot

```bash
uv run python -m src.evaluation.evaluate \
    --model whisper-large-v3 --language et \
    --test-sets cv17_validation cv17_test fleurs_test

uv run python -m src.evaluation.evaluate \
    --model parakeet-tdt-0.6b-v3 --language sl \
    --test-sets cv17_validation cv17_test fleurs_test
```

### Fine-tuned models

`--model-path` takes a local path or a Hub repo id.

```bash
# Released models
uv run python -m src.evaluation.evaluate \
    --model whisper-large-v3 --language et \
    --model-path yuriyvnv/whisper-large-v3-estonian \
    --test-sets cv17_test fleurs_test

uv run python -m src.evaluation.evaluate \
    --model parakeet-tdt-0.6b-v3 --language sl \
    --model-path yuriyvnv/parakeet-tdt-0.6b-slovenian \
    --test-sets cv17_test fleurs_test

# Your own runs (outputs of scripts/train_*.sh)
uv run python -m src.evaluation.evaluate \
    --model whisper-large-v3 --language et \
    --model-path ./results/whisper_finetune_et/whisper-large-v3-cv_synth_all_et-seed42 \
    --test-sets cv17_test fleurs_test

uv run python -m src.evaluation.evaluate \
    --model parakeet-tdt-0.6b-v3 --language et \
    --model-path ./results/parakeet_finetune_et/cv_synth_all_et_s42/parakeet-tdt-cv_synth_all_et-seed42.nemo \
    --test-sets cv17_test fleurs_test
```

### CLI arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | Yes | `whisper-large-v3` or `parakeet-tdt-0.6b-v3` |
| `--language` | Yes | `et`, `sl`, or `both` |
| `--test-sets` | Yes | One or more of `cv17_validation`, `cv17_test`, `fleurs_test` |
| `--model-path` | No | Local path or Hub repo id of a fine-tuned model (omit for zero-shot) |
| `--batch-size` | No | Inference batch size (default: 16) |
| `--output-dir` | No | Where to save results (default: `results/`) |

The script lowers its own process priority and caps itself at 30% of GPU memory so it can run next to a training job.

### Test sets

| Test set | Source | Estonian | Slovenian |
|----------|--------|---------|-----------|
| `cv17_validation` | `fixie-ai/common_voice_17_0` | 2,653 | 1,232 |
| `cv17_test` | `fixie-ai/common_voice_17_0` | 2,653 | 1,242 |
| `fleurs_test` | `google/fleurs` (`et_ee` / `sl_si`) | 893 | 834 |

All audio is resampled to 16 kHz.

### Output

One JSON per model, language and test set:

```
results/{whisperV3|parakeetV3}/{lang}/{model_label}_{lang}_{test_set}.json
```

`model_label` is the model name for zero-shot runs, otherwise the last component of `--model-path` without its extension. Each file holds WER, CER, Sub/Ins/Del counts and per-sentence reference, hypothesis and WER. WER and CER here are on raw text (whitespace strip only).

## `report_normalized.py`

Recomputes WER/CER from the per-sentence JSONs after lowercasing and punctuation removal, and prints raw vs normalized side by side.

```bash
uv run python -m src.evaluation.report_normalized [--results-dir results] [--save-xlsx]
```

## `significance.py`

Paired bootstrap resampling (Bisani & Ney, 2004) over sentences, B = 100,000, one-sided p-value from the sign-reversal fraction, 95% CI on the WER difference. Compares the five adaptation states pairwise on CV17 test and FLEURS test.

```bash
uv run python -m src.evaluation.significance --model parakeet-tdt-0.6b-v3 --language both
uv run python -m src.evaluation.significance --model whisper-large-v3 --language et --normalization normalized
```

It expects result files named after the run names used by the training scripts, e.g. `whisper-large-v3-cv_synth_all_et-seed42_et_cv17_test.json` and `parakeet-tdt-cv_only_sl-seed42_sl_fleurs_test.json`. Evaluating the outputs of `scripts/train_*.sh` as shown above produces exactly these names.
