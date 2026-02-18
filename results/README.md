# Evaluation Results

## Model: Parakeet-TDT-0.6B-v3

---

## Raw WER/CER (no text normalization)

### Estonian (et)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 26.93 | 6.26 | 27.19 | 6.42 | 39.14 | 7.54 |
| Fine-tuned CV only | 21.49 | 4.43 | 22.35 | 4.90 | 36.60 | 7.12 |
| Fine-tuned CV + Synth No Morph | 20.51 | 4.29 | 21.24 | 4.73 | 35.41 | 7.10 |
| Fine-tuned CV + Synth All | **20.18** | **4.21** | **21.03** | **4.64** | **35.29** | **7.06** |

### Slovenian (sl)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 58.78 | 31.16 | 50.23 | 24.51 | 40.18 | 11.53 |
| Fine-tuned CV only | 13.75 | 3.25 | 14.08 | 3.45 | 38.57 | 10.30 |
| Fine-tuned CV + Synth No Morph | 11.62 | 2.64 | 12.22 | 2.93 | 35.05 | 9.68 |
| Fine-tuned CV + Synth All | **11.30** | **2.61** | **11.56** | **2.87** | **34.71** | **9.75** |

---

## Normalized WER/CER (lowercase + punctuation removal)

Normalization: `jiwer.ToLowerCase()` + `jiwer.RemovePunctuation()` applied to both reference and hypothesis before computing metrics.

### Estonian (et)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 24.70 | 5.75 | 24.58 | 5.85 | 18.15 | 3.87 |
| Fine-tuned CV only | 19.24 | 3.99 | 19.74 | 4.38 | 14.29 | 3.34 |
| Fine-tuned CV + Synth No Morph | 18.29 | 3.85 | 18.69 | 4.22 | 12.70 | 3.26 |
| Fine-tuned CV + Synth All | **17.91** | **3.78** | **18.51** | **4.13** | **12.36** | **3.24** |

### Slovenian (sl)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 57.11 | 31.21 | 48.13 | 24.42 | 24.27 | 7.95 |
| Fine-tuned CV only | 11.63 | 2.90 | 11.69 | 3.07 | 22.36 | 6.70 |
| Fine-tuned CV + Synth No Morph | 9.21 | 2.23 | 9.68 | 2.48 | 17.90 | 6.03 |
| Fine-tuned CV + Synth All | **8.87** | **2.19** | **9.05** | **2.45** | **17.64** | **6.11** |

---

## Raw vs Normalized Comparison

Shows how much WER drops when punctuation and casing are removed. Large drops on FLEURS indicate the dataset has rich punctuation that the ASR model does not produce.

### Estonian (et)

| Model | Test Set | Raw WER | Norm WER | Diff |
|-------|----------|---------|----------|------|
| Zero-shot | CV17 test | 27.19 | 24.58 | -2.61 |
| Zero-shot | FLEURS test | 39.14 | 18.15 | -20.99 |
| CV only | CV17 test | 22.35 | 19.74 | -2.61 |
| CV only | FLEURS test | 36.60 | 14.29 | -22.31 |
| CV + Synth No Morph | CV17 test | 21.24 | 18.69 | -2.55 |
| CV + Synth No Morph | FLEURS test | 35.41 | 12.70 | -22.71 |
| CV + Synth All | CV17 test | 21.03 | 18.51 | -2.52 |
| CV + Synth All | FLEURS test | 35.29 | 12.36 | -22.93 |

### Slovenian (sl)

| Model | Test Set | Raw WER | Norm WER | Diff |
|-------|----------|---------|----------|------|
| Zero-shot | CV17 test | 50.23 | 48.13 | -2.10 |
| Zero-shot | FLEURS test | 40.18 | 24.27 | -15.91 |
| CV only | CV17 test | 14.08 | 11.69 | -2.39 |
| CV only | FLEURS test | 38.57 | 22.36 | -16.21 |
| CV + Synth No Morph | CV17 test | 12.22 | 9.68 | -2.54 |
| CV + Synth No Morph | FLEURS test | 35.05 | 17.90 | -17.15 |
| CV + Synth All | CV17 test | 11.56 | 9.05 | -2.51 |
| CV + Synth All | FLEURS test | 34.71 | 17.64 | -17.07 |

> **Note:** FLEURS references contain extensive punctuation and mixed casing that Parakeet does not produce, causing a ~15-23% WER inflation in raw scores. CV17 references are closer to ASR-style text with ~2.5% inflation. Normalized WER provides a fairer comparison of actual word recognition accuracy.

---

## WER Improvement Summary (Raw)

| Language | Comparison | CV17 Test | FLEURS Test |
|----------|-----------|-----------|-------------|
| Estonian | Zero-shot → CV only | -4.84 | -2.55 |
| Estonian | Zero-shot → CV + Synth No Morph | -5.95 | -3.73 |
| Estonian | Zero-shot → CV + Synth All | **-6.16** | **-3.85** |
| Estonian | CV only → CV + Synth No Morph | -1.11 | -1.19 |
| Estonian | CV only → CV + Synth All | **-1.32** | **-1.31** |
| Estonian | CV + Synth No Morph → CV + Synth All | -0.21 | -0.12 |
| Slovenian | Zero-shot → CV only | -36.15 | -1.61 |
| Slovenian | Zero-shot → CV + Synth No Morph | -38.01 | -5.13 |
| Slovenian | Zero-shot → CV + Synth All | **-38.67** | **-5.47** |
| Slovenian | CV only → CV + Synth No Morph | -1.86 | -3.52 |
| Slovenian | CV only → CV + Synth All | **-2.52** | **-3.86** |
| Slovenian | CV + Synth No Morph → CV + Synth All | -0.66 | -0.34 |

## WER Improvement Summary (Normalized)

| Language | Comparison | CV17 Test | FLEURS Test |
|----------|-----------|-----------|-------------|
| Estonian | Zero-shot → CV only | -4.84 | -3.86 |
| Estonian | Zero-shot → CV + Synth No Morph | -5.89 | -5.45 |
| Estonian | Zero-shot → CV + Synth All | **-6.07** | **-5.79** |
| Estonian | CV only → CV + Synth No Morph | -1.05 | -1.59 |
| Estonian | CV only → CV + Synth All | **-1.23** | **-1.93** |
| Estonian | CV + Synth No Morph → CV + Synth All | -0.18 | -0.34 |
| Slovenian | Zero-shot → CV only | -36.44 | -1.91 |
| Slovenian | Zero-shot → CV + Synth No Morph | -38.45 | -6.37 |
| Slovenian | Zero-shot → CV + Synth All | **-39.08** | **-6.63** |
| Slovenian | CV only → CV + Synth No Morph | -2.01 | -4.46 |
| Slovenian | CV only → CV + Synth All | **-2.64** | **-4.72** |
| Slovenian | CV + Synth No Morph → CV + Synth All | -0.63 | -0.26 |

---

## Statistical Significance (Estonian, Paired Bootstrap, n=100,000)

| Comparison | Test Set | Delta WER | 95% CI | p-value | Sig |
|-----------|----------|-----------|--------|---------|-----|
| Zero-shot vs CV-only | CV17 test | -4.84 | [-5.35, -4.33] | <1e-05 | *** |
| Zero-shot vs CV-only | FLEURS test | -2.55 | [-3.12, -1.98] | <1e-05 | *** |
| Zero-shot vs CV+Synth | CV17 test | -6.16 | [-6.68, -5.63] | <1e-05 | *** |
| Zero-shot vs CV+Synth | FLEURS test | -3.85 | [-4.43, -3.28] | <1e-05 | *** |
| CV-only vs CV+Synth | CV17 test | -1.31 | [-1.67, -0.96] | <1e-05 | *** |
| CV-only vs CV+Synth | FLEURS test | -1.30 | [-1.76, -0.85] | <1e-05 | *** |

Significance: *** p<0.001, ** p<0.01, * p<0.05

Statistical significance assessed using paired bootstrap resampling (Bisani & Ney, 2004) with 100,000 iterations.

---

## Test Set Sizes

| Test Set | Estonian | Slovenian |
|----------|---------|-----------|
| CV17 validation | 2,653 | 1,232 |
| CV17 test | 2,653 | 1,242 |
| FLEURS test | 893 | 834 |

## Training Details

- Base model: `nvidia/parakeet-tdt-0.6b-v3`
- Optimizer: AdamW (lr=5e-5, warmup 10%, cosine annealing)
- Batch size: 32
- Early stopping: patience 10 epochs on val_wer
- Seed: 42
