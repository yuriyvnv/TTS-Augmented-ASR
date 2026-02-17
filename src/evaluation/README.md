# Evaluation

Evaluate Whisper-large-v3 and Parakeet-TDT-0.6B-v3 on CommonVoice 17 and FLEURS test sets for Estonian and Slovenian.

## Usage

### Zero-shot evaluation

```bash
# Whisper on all test sets for Estonian
uv run python -m src.evaluation.evaluate \
    --model whisper-large-v3 --language et \
    --test-sets cv17_validation cv17_test fleurs_test

# Parakeet on all test sets for Slovenian
uv run python -m src.evaluation.evaluate \
    --model parakeet-tdt-0.6b-v3 --language sl \
    --test-sets cv17_validation cv17_test fleurs_test
```

### Fine-tuned model evaluation

```bash
# Whisper fine-tuned checkpoint
uv run python -m src.evaluation.evaluate \
    --model whisper-large-v3 --language et \
    --model-path ./results/whisper_et_cv_synth_all/checkpoint-best \
    --test-sets cv17_test fleurs_test

# Parakeet fine-tuned .nemo checkpoint
uv run python -m src.evaluation.evaluate \
    --model parakeet-tdt-0.6b-v3 --language et \
    --model-path ./results/parakeet_et/best.nemo \
    --test-sets cv17_test fleurs_test
```

## CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model` | Yes | `whisper-large-v3` or `parakeet-tdt-0.6b-v3` |
| `--language` | Yes | `et` (Estonian), `sl` (Slovenian), or `both` |
| `--test-sets` | Yes | One or more of: `cv17_validation`, `cv17_test`, `fleurs_test` |
| `--model-path` | No | Path to fine-tuned checkpoint (omit for zero-shot) |
| `--batch-size` | No | Inference batch size (default: 16) |
| `--output-dir` | No | Where to save results (default: `results/`) |

## Test Sets

| Test Set | Source | Estonian | Slovenian |
|----------|--------|---------|-----------|
| `cv17_validation` | `fixie-ai/common_voice_17_0` | 2,653 samples | 1,232 samples |
| `cv17_test` | `fixie-ai/common_voice_17_0` | 2,653 samples | 1,242 samples |
| `fleurs_test` | `google/fleurs` (`et_ee` / `sl_si`) | ~400 samples | ~400 samples |

All audio is cast to 16kHz before inference.

## Metrics

- **WER** (Word Error Rate): computed on raw text, no normalization
- **CER** (Character Error Rate): computed on raw text, no normalization
- **Sub/Ins/Del**: substitution, insertion, and deletion counts
- **Per-sentence WER**: for downstream statistical testing (bootstrap, Wilcoxon)

## Output

Results are saved as JSON files to `results/`:

```
results/
├── whisper-large-v3_et_cv17_test.json
├── whisper-large-v3_et_fleurs_test.json
├── parakeet-tdt-0.6b-v3_sl_cv17_test.json
└── ...
```

Each JSON file contains aggregate metrics, Sub/Ins/Del counts, and per-sentence results (reference, hypothesis, sentence WER).
