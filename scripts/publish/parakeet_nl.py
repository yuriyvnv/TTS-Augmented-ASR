"""
Push the fine-tuned Parakeet-TDT Dutch model to HuggingFace Hub.

Usage:
    uv run python scripts/push_parakeet_nl.py
"""

import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "yuriyvnv/parakeet-tdt-0.6b-dutch"
NEMO_PATH = Path("results/parakeet_finetune_nl/cv_synth_nl_s42/parakeet-tdt-cv_synth_nl-seed42.nemo")

README_CONTENT = """\
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
          name: Common Voice 17.0 (nl)
          type: fixie-ai/common_voice_17_0
          config: nl
          split: validation
        metrics:
          - type: wer
            value: 3.73
            name: Val WER
---

# Parakeet-TDT-0.6B Dutch

A Dutch automatic speech recognition (ASR) model fine-tuned from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## Model Details

| Property | Value |
|---|---|
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | FastConformer-TDT (600M params) |
| Language | Dutch (nl) |
| Val WER | **3.73%** |
| Input | 16 kHz mono audio |
| Output | Dutch text with punctuation and capitalization |
| License | CC-BY-4.0 |

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

## Citation

If you use this model, please cite the base Parakeet model:

```bibtex
@misc{parakeet-tdt-0.6b-v3,
  title={Parakeet TDT 0.6B v3},
  author={NVIDIA},
  year={2025},
  url={https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3}
}
```
"""


def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # Upload README
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=README_CONTENT.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        commit_message="Add model card",
    )

    # Upload .nemo model
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
