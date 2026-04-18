"""
Push the fine-tuned Parakeet-TDT Portuguese model to HuggingFace Hub.

Usage:
    uv run python scripts/push_parakeet_pt.py
"""

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "yuriyvnv/parakeet-tdt-0.6b-portuguese"
NEMO_PATH = Path("results/parakeet_finetune_pt/mixed_cv_synthetic_pt_s42/parakeet-tdt-mixed_cv_synthetic_pt-seed42.nemo")

README_CONTENT = """\
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
          name: Common Voice 17.0 (pt)
          type: fixie-ai/common_voice_17_0
          config: pt
          split: validation
        metrics:
          - type: wer
            value: 9.62
            name: Val WER
---

# Parakeet-TDT-0.6B Portuguese

A Portuguese automatic speech recognition (ASR) model fine-tuned from [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

## Model Details

| Property | Value |
|---|---|
| Base model | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | FastConformer-TDT (600M params) |
| Language | Portuguese (pt) |
| Val WER | **9.62%** |
| Input | 16 kHz mono audio |
| Output | Portuguese text with punctuation and capitalization |
| License | CC-BY-4.0 |

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
