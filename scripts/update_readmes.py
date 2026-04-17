"""
Update README model cards for Dutch and Portuguese Parakeet models
with test set results and improved metadata for visibility.

Usage:
    uv run python scripts/update_readmes.py
"""

from huggingface_hub import HfApi

api = HfApi()

# ── Dutch README ──────────────────────────────────────────────────────────────

NL_README = """\
---
language:
  - nl
license: cc-by-4.0
library_name: nemo
tags:
  - automatic-speech-recognition
  - speech
  - nemo
  - parakeet
  - fastconformer
  - tdt
  - dutch
  - nvidia
  - common-voice
  - synthetic-speech
  - fine-tuned
datasets:
  - fixie-ai/common_voice_17_0
  - yuriyvnv/synthetic_transcript_nl
base_model: nvidia/parakeet-tdt-0.6b-v3
pipeline_tag: automatic-speech-recognition
model-index:
  - name: parakeet-tdt-0.6b-dutch
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: Common Voice 17.0 (nl) - Validation
          type: fixie-ai/common_voice_17_0
          config: nl
          split: validation
        metrics:
          - type: wer
            value: 3.73
            name: Val WER
          - type: cer
            value: 1.02
            name: Val CER
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: Common Voice 17.0 (nl) - Test
          type: fixie-ai/common_voice_17_0
          config: nl
          split: test
        metrics:
          - type: wer
            value: 5.33
            name: Test WER
          - type: cer
            value: 1.46
            name: Test CER
---

# Parakeet-TDT-0.6B Dutch

A Dutch automatic speech recognition (ASR) model fine-tuned from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## Model Details

| Property | Value |
|---|---|
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | FastConformer-TDT (600M params) |
| Language | Dutch (nl) |
| Input | 16 kHz mono audio |
| Output | Dutch text with punctuation and capitalization |
| License | CC-BY-4.0 |

## Evaluation Results

Evaluated on [Common Voice 17.0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) Dutch splits (raw text, no normalization):

| Split | WER | CER | Samples |
|---|---|---|---|
| Validation | **3.73%** | 1.02% | 9,062 |
| Test | **5.33%** | 1.46% | 11,266 |

## Training

Fine-tuned on a combination of:

- **[Common Voice 17.0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)** (nl) -- human-recorded Dutch speech
- **[Synthetic Transcript NL](https://huggingface.co/datasets/yuriyvnv/synthetic_transcript_nl)** -- 34,898 synthetic Dutch speech samples generated with OpenAI TTS

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 (cosine annealing) |
| Warmup | 10% of total steps |
| Batch size | 64 |
| Precision | bf16-mixed |
| Gradient clipping | 1.0 |
| Early stopping | 10 epochs patience on val WER |
| Best epoch | 21 |

## Usage

### Installation

```bash
pip install nemo_toolkit[asr]
```

### Transcribe Audio

```python
import nemo.collections.asr as nemo_asr

# Load model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="yuriyvnv/parakeet-tdt-0.6b-dutch"
)

# Transcribe
output = asr_model.transcribe(["audio.wav"])
print(output[0].text)
```

### Transcribe with Timestamps

```python
output = asr_model.transcribe(["audio.wav"], timestamps=True)

for stamp in output[0].timestamp["segment"]:
    print(f"{stamp['start']:.1f}s - {stamp['end']:.1f}s : {stamp['segment']}")
```

### Long-Form Audio

For audio longer than 24 minutes, enable local attention:

```python
asr_model.change_attention_model(
    self_attention_model="rel_pos_local_attn",
    att_context_size=[256, 256],
)
output = asr_model.transcribe(["long_audio.wav"])
```

## Intended Use

This model is designed for transcribing Dutch speech to text. It works best on:
- Read speech and conversational Dutch
- Audio recorded at 16 kHz or higher
- Segments up to 24 minutes (or longer with local attention enabled)

## Limitations

- Trained primarily on European Portuguese-accented Dutch from Common Voice; performance may vary on regional dialects or heavily accented speech
- Synthetic training data was generated with OpenAI TTS voices, which may not fully represent natural speech variability
- Not suitable for real-time streaming without additional configuration
"""

# ── Portuguese README ─────────────────────────────────────────────────────────

PT_README = """\
---
language:
  - pt
license: cc-by-4.0
library_name: nemo
tags:
  - automatic-speech-recognition
  - speech
  - nemo
  - parakeet
  - fastconformer
  - tdt
  - portuguese
  - nvidia
  - common-voice
  - synthetic-speech
  - fine-tuned
datasets:
  - fixie-ai/common_voice_17_0
  - yuriyvnv/synthetic_transcript_pt
base_model: nvidia/parakeet-tdt-0.6b-v3
pipeline_tag: automatic-speech-recognition
model-index:
  - name: parakeet-tdt-0.6b-portuguese
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: Common Voice 17.0 (pt) - Validation
          type: fixie-ai/common_voice_17_0
          config: pt
          split: validation
        metrics:
          - type: wer
            value: 9.62
            name: Val WER
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: Common Voice 17.0 (pt) - Test
          type: fixie-ai/common_voice_17_0
          config: pt
          split: test
        metrics:
          - type: wer
            value: 10.71
            name: Test WER
          - type: cer
            value: 2.69
            name: Test CER
---

# Parakeet-TDT-0.6B Portuguese

A Portuguese automatic speech recognition (ASR) model fine-tuned from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## Model Details

| Property | Value |
|---|---|
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | FastConformer-TDT (600M params) |
| Language | Portuguese (pt) |
| Input | 16 kHz mono audio |
| Output | Portuguese text with punctuation and capitalization |
| License | CC-BY-4.0 |

## Evaluation Results

Evaluated on [Common Voice 17.0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) Portuguese splits (raw text, no normalization):

| Split | WER | CER | Samples |
|---|---|---|---|
| Validation | **9.62%** | -- | 9,464 |
| Test | **10.71%** | 2.69% | 9,467 |

## Training

Fine-tuned on the `mixed_cv_synthetic` configuration from [yuriyvnv/synthetic_transcript_pt](https://huggingface.co/datasets/yuriyvnv/synthetic_transcript_pt):

- **[Common Voice 17.0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)** (pt) -- 21,968 human-recorded Portuguese speech samples
- **Synthetic samples** -- 19,181 high-quality synthetic Portuguese speech samples generated with OpenAI TTS (filtered by audio-text similarity > 0.5)
- **Total training set**: 41,149 samples

Validation and test sets use Common Voice 17.0 Portuguese exclusively (no synthetic contamination).

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 (cosine annealing) |
| Warmup | 10% of total steps |
| Batch size | 64 |
| Precision | bf16-mixed |
| Gradient clipping | 1.0 |
| Early stopping | 10 epochs patience on val WER |
| Best epoch | 15 |

## Usage

### Installation

```bash
pip install nemo_toolkit[asr]
```

### Transcribe Audio

```python
import nemo.collections.asr as nemo_asr

# Load model
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="yuriyvnv/parakeet-tdt-0.6b-portuguese"
)

# Transcribe
output = asr_model.transcribe(["audio.wav"])
print(output[0].text)
```

### Transcribe with Timestamps

```python
output = asr_model.transcribe(["audio.wav"], timestamps=True)

for stamp in output[0].timestamp["segment"]:
    print(f"{stamp['start']:.1f}s - {stamp['end']:.1f}s : {stamp['segment']}")
```

### Long-Form Audio

For audio longer than 24 minutes, enable local attention:

```python
asr_model.change_attention_model(
    self_attention_model="rel_pos_local_attn",
    att_context_size=[256, 256],
)
output = asr_model.transcribe(["long_audio.wav"])
```

## Intended Use

This model is designed for transcribing Portuguese speech to text. It works best on:
- Read speech and conversational Portuguese
- Audio recorded at 16 kHz or higher
- Segments up to 24 minutes (or longer with local attention enabled)

## Limitations

- Trained on Common Voice data which is predominantly European Portuguese; performance may differ on Brazilian Portuguese
- Synthetic training data was generated with OpenAI TTS voices, which may not fully represent natural speech variability
- Not suitable for real-time streaming without additional configuration
"""


def main():
    # Update Dutch
    print("Updating yuriyvnv/parakeet-tdt-0.6b-dutch README...")
    api.upload_file(
        path_or_fileobj=NL_README.encode(),
        path_in_repo="README.md",
        repo_id="yuriyvnv/parakeet-tdt-0.6b-dutch",
        commit_message="Add test set results (WER/CER) and improve metadata",
    )
    print("  Done.")

    # Update Portuguese
    print("Updating yuriyvnv/parakeet-tdt-0.6b-portuguese README...")
    api.upload_file(
        path_or_fileobj=PT_README.encode(),
        path_in_repo="README.md",
        repo_id="yuriyvnv/parakeet-tdt-0.6b-portuguese",
        commit_message="Add test set results (WER/CER) and improve metadata",
    )
    print("  Done.")

    print("\nBoth model cards updated!")


if __name__ == "__main__":
    main()
