# TTS-Augmented-ASR

Code and results for the paper **"Synthetic speech augmentation for low-resource Estonian and Slovenian ASR: Comparing Parakeet-TDT and Whisper"** (Y. Perezhohin, M. Castelli, A. Popovič), *Machine Learning with Applications* 26 (2026) 101043.

[Paper](https://www.sciencedirect.com/science/article/pii/S2666827026002082) · [DOI](https://doi.org/10.1016/j.mlwa.2026.101043) · [Models on Hugging Face](https://huggingface.co/yuriyvnv/models?sort=downloads) · [Dataset](https://huggingface.co/datasets/yuriyvnv/synthetic_asr_et_sl)

The repository covers the full pipeline: LLM text generation with LLM-as-judge validation, TTS synthesis, dataset building, fine-tuning of Whisper-large-v3 and Parakeet-TDT-0.6B-v3, and evaluation on CommonVoice 17 and FLEURS.

## Released models

| Model | Language | Base model | Training data |
|---|---|---|---|
| [yuriyvnv/parakeet-tdt-0.6b-estonian](https://huggingface.co/yuriyvnv/parakeet-tdt-0.6b-estonian) | Estonian | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CV17 + synthetic (all) |
| [yuriyvnv/parakeet-tdt-0.6b-slovenian](https://huggingface.co/yuriyvnv/parakeet-tdt-0.6b-slovenian) | Slovenian | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CV17 + synthetic (unfiltered) |
| [yuriyvnv/whisper-large-v3-estonian](https://huggingface.co/yuriyvnv/whisper-large-v3-estonian) | Estonian | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | CV17 + synthetic (all) |
| [yuriyvnv/whisper-large-v3-slovenian](https://huggingface.co/yuriyvnv/whisper-large-v3-slovenian) | Slovenian | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | CV17 + synthetic (no morphological) |

For each model and language, the released model is the training configuration with the lowest CommonVoice 17 validation WER.

**Dataset:** [yuriyvnv/synthetic_asr_et_sl](https://huggingface.co/datasets/yuriyvnv/synthetic_asr_et_sl), with one config per training condition (see [Training conditions](#training-conditions)). Validation and test splits are always the real CV17 dev/test sets.

### Using the models

```python
# Parakeet (NeMo)
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("yuriyvnv/parakeet-tdt-0.6b-estonian")
print(model.transcribe(["audio.wav"])[0].text)
```

```python
# Whisper (transformers)
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="yuriyvnv/whisper-large-v3-slovenian")
print(asr("audio.wav", generate_kwargs={"language": "slovenian", "task": "transcribe"})["text"])
```

## Repository layout

```
├── src/
│   ├── data_pipeline/          # download → text generation → TTS → HF dataset
│   │   ├── download_data.py
│   │   ├── text_diversification.py
│   │   ├── tts_synthesis.py
│   │   └── prepare_and_push_dataset.py
│   ├── training/               # fine-tuning
│   │   ├── train_whisper.py
│   │   └── train_parakeet.py
│   └── evaluation/             # WER/CER and significance tests
│       ├── evaluate.py
│       ├── report_normalized.py
│       └── significance.py
├── prompts/                    # generation, validation and regeneration prompts
├── scripts/                    # setup and training launchers with the paper's settings
├── results/                    # paper results: tables, per-sentence outputs, training logs
├── pyproject.toml / uv.lock
└── .env.example
```

Each `src/` subfolder has its own README with full CLI details.

## Setup

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for training (experiments were run on a single H100 80GB).

```bash
uv sync
cp .env.example .env      # fill in the keys you need
```

- `OPENAI_API_KEY` is only needed to regenerate the synthetic data.
- `HF_TOKEN` (or `hf auth login`) and `WANDB_API_KEY` (or `wandb login`) are used for training.

On a fresh GPU machine, `./scripts/setup_gpu.sh` installs uv, syncs dependencies and runs the logins.

## Reproducing the experiments

### 1. Synthetic data (optional)

The finished dataset is on the Hub, so this step is only needed to rebuild it. It calls the OpenAI API (`gpt-5-mini` for text, `gpt-4o-mini-tts` for speech).

```bash
uv run python -m src.data_pipeline.download_data --language et
uv run python -m src.data_pipeline.text_diversification --language et --phase all \
    --seeds data/seeds_et.txt --test-texts data/test_texts_et.txt
uv run python -m src.data_pipeline.tts_synthesis --language et
uv run python -m src.data_pipeline.prepare_and_push_dataset --repo-id <your-user>/<dataset> --dry-run
```

See [src/data_pipeline/README.md](src/data_pipeline/README.md).

### 2. Fine-tuning

Both launchers use the paper's hyperparameters and run all 8 language/config pairs by default, or a single pair:

```bash
./scripts/train_whisper.sh                       # all 8 runs
./scripts/train_parakeet.sh et cv_synth_all_et   # one run
HUB_REPO=<your-user>/<repo> ./scripts/train_parakeet.sh et cv_synth_all_et   # also upload
```

See [src/training/README.md](src/training/README.md).

### 3. Evaluation

```bash
# Zero-shot
uv run python -m src.evaluation.evaluate --model whisper-large-v3 --language both \
    --test-sets cv17_test fleurs_test

# A released model (local paths work too)
uv run python -m src.evaluation.evaluate --model parakeet-tdt-0.6b-v3 --language et \
    --model-path yuriyvnv/parakeet-tdt-0.6b-estonian --test-sets cv17_test fleurs_test

# Raw vs normalized WER, and paired bootstrap tests (B = 100,000)
uv run python -m src.evaluation.report_normalized
uv run python -m src.evaluation.significance --model parakeet-tdt-0.6b-v3 --language both
```

See [src/evaluation/README.md](src/evaluation/README.md).

## Experimental setup

### Training conditions

| Config | Train split |
|---|---|
| `cv_only_{lang}` | CommonVoice 17 train only |
| `cv_synth_all_{lang}` | CV17 train + validated synthetic data (paraphrase, domain, morphological) |
| `cv_synth_no_morph_{lang}` | CV17 train + validated synthetic data without the morphological category |
| `cv_synth_unfiltered_{lang}` | CV17 train + synthetic data before LLM-as-judge filtering |

Each model is also evaluated zero-shot, giving five adaptation states per model and language.

### Hyperparameters

All runs are full fine-tuning with AdamW, peak learning rate 5e-5, 10% linear warmup and seed 42.

| | Whisper-large-v3 | Parakeet-TDT-0.6B-v3 |
|---|---|---|
| Batch size | 64 x 2 accumulation = 128 | 32 |
| Schedule | linear decay, 5 epochs | cosine annealing, up to 100 epochs |
| Model selection | lowest validation loss (eval every 25 steps) | lowest validation WER, early stopping patience 10 |
| Precision | bf16 | bf16-mixed |

### Evaluation data

| | Estonian | Slovenian |
|---|---|---|
| CommonVoice 17 test ([fixie-ai/common_voice_17_0](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)) | 2,653 | 1,242 |
| FLEURS test ([google/fleurs](https://huggingface.co/datasets/google/fleurs), `et_ee` / `sl_si`) | 893 | 834 |

WER and CER are computed with `jiwer`, both on raw text and after lowercasing and punctuation removal.

## Results

[`results/README.md`](results/README.md) has the paper's result tables: raw and normalized WER/CER for every model, language, adaptation state and test set, plus the paired bootstrap significance tests. The underlying files are:

- `results/{parakeetV3,whisperV3}/{et,sl}/*.json`: one file per model, language and test set (`cv17_validation`, `cv17_test`, `fleurs_test`), with aggregate metrics and per-sentence reference, hypothesis and WER, as written by `src.evaluation.evaluate`
- `results/normalized_results.xlsx`: raw vs normalized WER/CER for all 60 evaluations (`src.evaluation.report_normalized`)
- `results/training_logs/`: Weights & Biases training histories for the 16 fine-tuning runs

The tables can be regenerated from the JSONs with `src.evaluation.report_normalized` and `src.evaluation.significance`.

## Citation

```bibtex
@article{PEREZHOHIN2026101043,
  title   = {Synthetic speech augmentation for low-resource Estonian and Slovenian ASR: Comparing Parakeet-TDT and Whisper},
  journal = {Machine Learning with Applications},
  volume  = {26},
  pages   = {101043},
  year    = {2026},
  issn    = {2666-8270},
  doi     = {10.1016/j.mlwa.2026.101043},
  url     = {https://www.sciencedirect.com/science/article/pii/S2666827026002082},
  author  = {Yuriy Perezhohin and Mauro Castelli and Aleš Popovič},
}
```

## License

The code is released under the [MIT License](LICENSE). The dataset is CC BY 4.0. The released models keep their base models' terms: Parakeet models are CC BY 4.0 and Whisper models are Apache-2.0 (see each model card).
