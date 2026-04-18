"""
Push the fine-tuned Parakeet-TDT Polish model to HuggingFace Hub.

Usage:
    uv run python scripts/push_parakeet_pl.py
"""

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "yuriyvnv/parakeet-tdt-0.6b-polish"
NEMO_PATH = Path("results/parakeet_finetune_pl/bigos_cased_pl_s42/parakeet-tdt-bigos_cased_pl-seed42.nemo")

README_CONTENT = """\
---
language:
  - pl
license: cc-by-4.0
library_name: nemo
tags:
  - automatic-speech-recognition
  - speech
  - nemo
  - parakeet
  - fastconformer
  - tdt
  - polish
  - nvidia
  - common-voice
  - bigos
  - fine-tuned
datasets:
  - amu-cai/pl-asr-bigos-v2
  - fixie-ai/common_voice_17_0
base_model: nvidia/parakeet-tdt-0.6b-v3
pipeline_tag: automatic-speech-recognition
model-index:
  - name: parakeet-tdt-0.6b-polish
    results:
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: Common Voice 17.0 (pl) - Validation
          type: fixie-ai/common_voice_17_0
          config: pl
          split: validation
        metrics:
          - type: wer
            value: 6.07
            name: Val WER
      - task:
          type: automatic-speech-recognition
          name: Speech Recognition
        dataset:
          name: Common Voice 17.0 (pl) - Test
          type: fixie-ai/common_voice_17_0
          config: pl
          split: test
        metrics:
          - type: wer
            value: 11.81
            name: Test WER
          - type: cer
            value: 2.72
            name: Test CER
---

# Parakeet-TDT-0.6B Polish

A Polish automatic speech recognition (ASR) model fine-tuned from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## Model Details

| Property | Value |
|---|---|
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | FastConformer-TDT (600M params) |
| Language | Polish (pl) |
| Input | 16 kHz mono audio |
| Output | Polish text with punctuation and capitalization |
| License | CC-BY-4.0 |

## Evaluation Results

Evaluated on [Common Voice 17.0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) Polish (raw text, no normalization):

| Split | WER | CER | Samples |
|---|---|---|---|
| Validation | **6.07%** | -- | -- |
| Test | **11.81%** | 2.72% | 9,230 |

## Training

Fine-tuned on a curated subset of the [BIGOS v2](https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2) benchmark, filtered to retain only sources with proper casing and punctuation:

- **[Common Voice 15](https://commonvoice.mozilla.org/)** -- 19,119 human-recorded Polish speech samples
- **[M-AILABS / LibriVox](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)** -- 11,834 read Polish audiobook samples
- **[PolyAI Minds14](https://huggingface.co/datasets/PolyAI/minds14)** -- 462 Polish banking dialog samples
- **Total training set**: ~31,415 samples

Validation uses the BIGOS v2 validation split (same source filtering). Test evaluation uses Common Voice 17.0 Polish (independent test set).

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 (cosine annealing) |
| Warmup | 10% of total steps |
| Batch size | 32 |
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
    model_name="yuriyvnv/parakeet-tdt-0.6b-polish"
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

This model is designed for transcribing Polish speech to text. It works best on:
- Read speech and conversational Polish
- Audio recorded at 16 kHz or higher
- Segments up to 24 minutes (or longer with local attention enabled)

## Limitations

- Training data is sourced from read speech (audiobooks, Common Voice read prompts) and short banking dialogs; performance may differ on spontaneous or heavily accented speech
- The model preserves punctuation and capitalization as seen in training data
- Not suitable for real-time streaming without additional configuration
"""


def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=README_CONTENT.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        commit_message="Add model card",
    )

    print(f"Uploading {NEMO_PATH.name} ({NEMO_PATH.stat().st_size / 1e9:.1f} GB)...")
    api.upload_file(
        path_or_fileobj=str(NEMO_PATH),
        path_in_repo=NEMO_PATH.name,
        repo_id=REPO_ID,
        commit_message="Upload fine-tuned model",
    )

    print(f"\nDone! Model available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
