# Evaluation Results

---

## Model: Parakeet-TDT-0.6B-v3

### Raw WER/CER (no text normalization)

#### Estonian (et)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 26.93 | 6.26 | 27.19 | 6.42 | 39.14 | 7.54 |
| Fine-tuned CV only | 21.49 | 4.43 | 22.35 | 4.90 | 36.60 | 7.12 |
| Fine-tuned CV + Synth No Morph | 20.51 | 4.29 | 21.24 | 4.73 | 35.41 | 7.10 |
| Fine-tuned CV + Synth All | **20.18** | **4.21** | **21.03** | **4.64** | **35.29** | **7.06** |

#### Slovenian (sl)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 58.78 | 31.16 | 50.23 | 24.51 | 40.18 | 11.53 |
| Fine-tuned CV only | 13.75 | 3.25 | 14.08 | 3.45 | 38.57 | 10.30 |
| Fine-tuned CV + Synth No Morph | 11.62 | 2.64 | 12.22 | 2.93 | 35.05 | 9.68 |
| Fine-tuned CV + Synth All | **11.30** | **2.61** | **11.56** | **2.87** | **34.71** | **9.75** |

### Normalized WER/CER (lowercase + punctuation removal)

#### Estonian (et)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 24.70 | 5.75 | 24.58 | 5.85 | 18.15 | 3.87 |
| Fine-tuned CV only | 19.24 | 3.99 | 19.74 | 4.38 | 14.29 | 3.34 |
| Fine-tuned CV + Synth No Morph | 18.29 | 3.85 | 18.69 | 4.22 | 12.70 | 3.26 |
| Fine-tuned CV + Synth All | **17.91** | **3.78** | **18.51** | **4.13** | **12.36** | **3.24** |

#### Slovenian (sl)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 57.11 | 31.21 | 48.13 | 24.42 | 24.27 | 7.95 |
| Fine-tuned CV only | 11.63 | 2.90 | 11.69 | 3.07 | 22.36 | 6.70 |
| Fine-tuned CV + Synth No Morph | 9.21 | 2.23 | 9.68 | 2.48 | 17.90 | 6.03 |
| Fine-tuned CV + Synth All | **8.87** | **2.19** | **9.05** | **2.45** | **17.64** | **6.11** |

### Statistical Significance — Parakeet (Paired Bootstrap, n=100,000)

#### Estonian

| Comparison | Test Set | Delta WER | 95% CI | p-value | Sig |
|-----------|----------|-----------|--------|---------|-----|
| Zero-shot vs CV-only | CV17 test | -4.84 | [-5.35, -4.33] | <1e-05 | *** |
| Zero-shot vs CV-only | FLEURS test | -2.55 | [-3.12, -1.98] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | CV17 test | -6.16 | [-6.68, -5.63] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | FLEURS test | -3.85 | [-4.43, -3.28] | <1e-05 | *** |
| CV-only vs CV+Synth All | CV17 test | -1.31 | [-1.67, -0.96] | <1e-05 | *** |
| CV-only vs CV+Synth All | FLEURS test | -1.30 | [-1.76, -0.85] | <1e-05 | *** |

---

## Model: Whisper-large-v3

### Raw WER/CER (no text normalization)

#### Estonian (et)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 33.18 | 7.40 | 34.40 | 8.28 | — | — |
| Fine-tuned CV + Synth No Morph | 25.87 | 5.63 | 26.50 | 6.20 | — | — |
| Fine-tuned CV + Synth All | **25.25** | **5.63** | **26.46** | **6.24** | — | — |

#### Slovenian (sl)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 21.36 | 5.46 | 21.20 | 5.59 | 37.02 | 9.37 |
| Fine-tuned CV only | 19.87 | 4.75 | 19.31 | 4.61 | 46.79 | 14.50 |
| Fine-tuned CV + Synth No Morph | 15.08 | 3.60 | **15.65** | **3.88** | **40.46** | **11.98** |
| Fine-tuned CV + Synth All | **15.31** | **3.70** | 16.40 | 4.14 | 41.11 | 11.77 |

> **Note:** Whisper fine-tuning on CV data degrades FLEURS performance for Slovenian. Zero-shot Whisper (37.02%) outperforms all fine-tuned variants on FLEURS raw WER. This is because Whisper's pre-trained model handles FLEURS-style punctuation and casing, which fine-tuning on CV-style data erodes.

### Normalized WER/CER (lowercase + punctuation removal)

#### Estonian (et)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 30.80 | 6.81 | 31.52 | 7.59 | — | — |
| Fine-tuned CV + Synth No Morph | 23.70 | 5.13 | 24.01 | 5.64 | — | — |
| Fine-tuned CV + Synth All | **23.11** | **5.15** | **24.00** | **5.69** | — | — |

#### Slovenian (sl)

| Model | CV17 Val WER | CV17 Val CER | CV17 Test WER | CV17 Test CER | FLEURS Test WER | FLEURS Test CER |
|-------|-------------|-------------|--------------|--------------|----------------|----------------|
| Zero-shot (baseline) | 18.45 | 4.77 | 18.60 | 5.03 | **19.18** | **5.45** |
| Fine-tuned CV only | 17.70 | 4.34 | 16.85 | 4.21 | 32.52 | 10.93 |
| Fine-tuned CV + Synth No Morph | **12.98** | **3.24** | **13.24** | **3.48** | 24.52 | 8.32 |
| Fine-tuned CV + Synth All | 13.02 | 3.30 | 14.00 | 3.73 | 25.12 | 8.05 |

> **Note:** Even with normalization, Whisper zero-shot (19.18%) outperforms fine-tuned models on FLEURS for Slovenian. However, on CV17 test the synthetic augmentation provides large improvements: -5.36% (CV-only→Synth No Morph) and -2.85% (CV-only→Synth All).

### Statistical Significance — Whisper Raw WER (Paired Bootstrap, n=100,000)

#### Estonian

| Comparison | Test Set | Delta WER | 95% CI | p-value | Sig |
|-----------|----------|-----------|--------|---------|-----|
| Zero-shot vs CV+Synth All | CV17 test | -7.94 | [-8.78, -7.21] | <1e-05 | *** |
| Zero-shot vs CV+Synth No Morph | CV17 test | -7.90 | [-8.73, -7.17] | <1e-05 | *** |

#### Slovenian

| Comparison | Test Set | Delta WER | 95% CI | p-value | Sig |
|-----------|----------|-----------|--------|---------|-----|
| Zero-shot vs CV-only | CV17 test | -1.89 | [-3.28, -0.50] | 0.00401 | ** |
| Zero-shot vs CV+Synth No Morph | CV17 test | -5.54 | [-6.86, -4.24] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | CV17 test | -4.79 | [-6.11, -3.48] | <1e-05 | *** |
| CV-only vs CV+Synth No Morph | CV17 test | -3.66 | [-4.74, -2.58] | <1e-05 | *** |
| CV-only vs CV+Synth All | CV17 test | -2.91 | [-4.02, -1.80] | <1e-05 | *** |
| CV+Synth No Morph vs CV+Synth All | CV17 test | +0.75 | [-0.20, +1.72] | 0.06469 | n.s. |
| Zero-shot vs CV-only | FLEURS test | +9.76 | [+8.85, +10.68] | <1e-05 | *** |
| Zero-shot vs CV+Synth No Morph | FLEURS test | +3.43 | [+2.68, +4.19] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | FLEURS test | +4.08 | [+3.36, +4.82] | <1e-05 | *** |
| CV-only vs CV+Synth No Morph | FLEURS test | -6.33 | [-7.18, -5.49] | <1e-05 | *** |
| CV-only vs CV+Synth All | FLEURS test | -5.68 | [-6.47, -4.89] | <1e-05 | *** |
| CV+Synth No Morph vs CV+Synth All | FLEURS test | +0.65 | [-0.01, +1.32] | 0.02651 | * |

> **Note:** FLEURS deltas are positive for Zero-shot vs fine-tuned (fine-tuning hurts FLEURS performance). CV17 test deltas are all negative (fine-tuning helps in-domain). CV+Synth No Morph vs CV+Synth All is not significant on CV17 (p=0.065) and marginally significant on FLEURS (p=0.027).

### Statistical Significance — Whisper Normalized WER (Paired Bootstrap, n=100,000)

#### Estonian

| Comparison | Test Set | Delta WER | 95% CI | p-value | Sig |
|-----------|----------|-----------|--------|---------|-----|
| Zero-shot vs CV+Synth No Morph | CV17 test | -7.51 | [-8.34, -6.79] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | CV17 test | -7.52 | [-8.36, -6.79] | <1e-05 | *** |

#### Slovenian

| Comparison | Test Set | Delta WER | 95% CI | p-value | Sig |
|-----------|----------|-----------|--------|---------|-----|
| Zero-shot vs CV-only | CV17 test | -1.74 | [-3.10, -0.40] | 0.00612 | ** |
| Zero-shot vs CV+Synth No Morph | CV17 test | -5.36 | [-6.63, -4.11] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | CV17 test | -4.60 | [-5.88, -3.32] | <1e-05 | *** |
| CV-only vs CV+Synth No Morph | CV17 test | -3.62 | [-4.66, -2.58] | <1e-05 | *** |
| CV-only vs CV+Synth All | CV17 test | -2.85 | [-3.92, -1.78] | <1e-05 | *** |
| CV+Synth No Morph vs CV+Synth All | CV17 test | +0.77 | [-0.18, +1.72] | 0.05832 | n.s. |
| Zero-shot vs CV-only | FLEURS test | +13.34 | [+12.34, +14.35] | <1e-05 | *** |
| Zero-shot vs CV+Synth No Morph | FLEURS test | +5.33 | [+4.48, +6.18] | <1e-05 | *** |
| Zero-shot vs CV+Synth All | FLEURS test | +5.94 | [+5.13, +6.78] | <1e-05 | *** |
| CV-only vs CV+Synth No Morph | FLEURS test | -8.01 | [-8.96, -7.07] | <1e-05 | *** |
| CV-only vs CV+Synth All | FLEURS test | -7.40 | [-8.29, -6.52] | <1e-05 | *** |
| CV+Synth No Morph vs CV+Synth All | FLEURS test | +0.61 | [-0.12, +1.34] | 0.05060 | n.s. |

> **Note:** With normalization, CV+Synth No Morph vs CV+Synth All remains not significant on both CV17 (p=0.058) and FLEURS (p=0.051). The morphological augmentation does not provide additional benefit for Whisper.

Significance: *** p<0.001, ** p<0.01, * p<0.05. Statistical significance assessed using paired bootstrap resampling (Bisani & Ney, 2004) with 100,000 iterations.

---

## WER Improvement Summary — Parakeet (Raw)

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

## WER Improvement Summary — Whisper (Raw)

| Language | Comparison | CV17 Test | FLEURS Test |
|----------|-----------|-----------|-------------|
| Estonian | Zero-shot → CV + Synth No Morph | -7.90 | — |
| Estonian | Zero-shot → CV + Synth All | **-7.94** | — |
| Slovenian | Zero-shot → CV only | -1.89 | +9.76 |
| Slovenian | Zero-shot → CV + Synth No Morph | **-5.55** | +3.44 |
| Slovenian | Zero-shot → CV + Synth All | -4.80 | +4.09 |
| Slovenian | CV only → CV + Synth No Morph | **-3.66** | **-6.33** |
| Slovenian | CV only → CV + Synth All | -2.91 | -5.68 |
| Slovenian | CV + Synth No Morph → CV + Synth All | +0.75 (n.s.) | +0.65 (*) |

---

## Test Set Sizes

| Test Set | Estonian | Slovenian |
|----------|---------|-----------|
| CV17 validation | 2,653 | 1,232 |
| CV17 test | 2,653 | 1,242 |
| FLEURS test | 893 | 834 |

## Training Details

### Parakeet-TDT-0.6B-v3
- Base model: `nvidia/parakeet-tdt-0.6b-v3`
- Optimizer: AdamW (lr=5e-5, warmup 10%, cosine annealing)
- Batch size: 32
- Early stopping: patience 10 epochs on val_wer
- Seed: 42

### Whisper-large-v3
- Base model: `openai/whisper-large-v3`
- Optimizer: AdamW fused (lr=5e-5, warmup 10%)
- Effective batch size: 128 (64 x 2 gradient accumulation)
- Epochs: 5
- Best model selected by eval_loss
- Seed: 42

---

Normalization: `jiwer.ToLowerCase()` + `jiwer.RemovePunctuation()` applied to both reference and hypothesis before computing metrics.
