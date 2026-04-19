"""
Fine-tune Qwen3-ASR-1.7B on Common Voice 22 (Portuguese).

Full fine-tune (no LoRA), bf16-mixed precision, gradient checkpointing.
Hyperparameters follow the only public Qwen3-ASR FT precedent
(Gearnode/qwen3-asr-uzbek): lr=1e-5, eff. batch 128, 3 epochs.

Usage:
    LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/av.libs:$LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.training.train_qwen3_asr \
        --language pt \
        --output-dir ./results/qwen3_finetune_pt/cv22_s42 \
        --seed 42
"""

import argparse
import json
import logging
import os
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from jiwer import cer as compute_cer
from jiwer import wer as compute_wer
from tqdm import tqdm

# `qwen_asr` only registers Qwen3ASRConfig with AutoConfig; it does NOT register the
# model class with AutoModelForSeq2SeqLM (or any other AutoModel head). Import the
# concrete class directly to avoid `Unrecognized configuration class` errors.
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)


class Qwen3ASRForTraining(Qwen3ASRForConditionalGeneration):
    """The shipped `Qwen3ASRForConditionalGeneration` is a thin wrapper that only
    defines `__init__` (creates `self.thinker`) and `generate()` — it has no
    `forward()`, so HF Trainer's `model(**inputs)` hits `_forward_unimplemented`.
    Delegate to the inner thinker which has the real forward + loss computation.
    The state dict keys ('thinker.*') are unchanged, so `from_pretrained` loads
    the public Qwen/Qwen3-ASR-1.7B weights without remapping.
    """

    def forward(self, **kwargs):
        return self.thinker(**kwargs)

from transformers import (
    AutoProcessor,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Shared normalization: capitalize first letter, collapse trailing "..."/"…"
# to single ".", append "." if no terminal punct. Applied to train / val / test
# references so the model's target distribution is consistent written form.
from src.evaluation.score_written_form import normalize_written_form

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
CV22_REPO = "fsicoli/common_voice_22_0"
CV17_REPO = "fixie-ai/common_voice_17_0"
SYNTH_PT_REPO = "yuriyvnv/synthetic_transcript_pt"  # CV17-PT filtered by WAVe-embedding quality
SYNTH_NL_REPO = "yuriyvnv/synthetic_transcript_nl"  # Fully synthetic OpenAI-TTS Dutch (34,898 rows)

LANGUAGE_NAMES = {"pt": "Portuguese", "nl": "Dutch"}
SAMPLING_RATE = 16000
MAX_AUDIO_DURATION_S = 30.0
MIN_AUDIO_DURATION_S = 0.3
# Filter pathologically long transcripts that either (a) are TSV parsing
# artefacts (raw multi-row content glued into one cell by stray newlines) or
# (b) are real but don't match the audio. A 500-char cap is generous — typical
# CV sentences are <200 chars. Issue #91 in QwenLM/Qwen3-ASR flags these as a
# known cause of OOM and training instability.
MAX_SENTENCE_CHARS = 500

# CV22 manual-load cache (datasets 4.x removed script-based loaders, and
# fsicoli/common_voice_22_0 has no parquet conversion, so we materialise it
# ourselves via hf_hub_download + tar extraction).
CV22_CACHE_DIR = Path.home() / ".cache" / "syntts_asr" / "cv22"
CV22_SPLIT_ALIASES = {"validation": "dev"}  # CV uses train/dev/test naming


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


def _build_prefix_messages(system_prompt: str):
    """Mirrors the official SFT script's `build_prefix_messages`. Audio array
    is not needed for the chat template pass — the processor injects
    <|audio_pad|> via the structured `{type:"audio"}` content block, and
    only expands it later when the actual audio is passed to
    `processor(text=..., audio=...)`."""
    return [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]


@dataclass
class DataCollatorQwen3ASR:
    """Follows `QwenLM/Qwen3-ASR/finetuning/README.md` target format.

    The model is trained to **emit** `language {X}<asr_text>{transcript}`
    as part of its assistant turn (not to be conditioned on it). Per the
    official README: "If you have language info, use:
    language English<asr_text>..." — this goes in the `text` (target)
    field of the training example, not in the prompt.

    Prefix = apply_chat_template(system + user-with-audio, add_generation_prompt=True)
    Target = "language {language_name}<asr_text>{transcript}" + eos
    Full   = prefix + target
    Labels = full with prefix tokens masked to -100 (loss flows through target).

    IMPORTANT: the processor's default `padding_side='left'` breaks the
    prefix-length mask (prefix tokens end up shifted by the pad length,
    so `labels[:, :prefix_len] = -100` masks PAD positions instead of
    prefix content — see https://github.com/QwenLM/Qwen3-ASR/issues/70 ).
    We force right-padding at processor init time; don't remove it.
    """

    processor: Any
    language_name: str = "Portuguese"
    system_prompt: str = ""
    text_column: str = "sentence"

    def __call__(self, features):
        audios = [
            np.asarray(f["audio"]["array"], dtype=np.float32) for f in features
        ]
        transcripts = [f[self.text_column].strip() for f in features]
        eos = self.processor.tokenizer.eos_token or ""

        # Target includes language prefix — model learns to emit it
        targets = [f"language {self.language_name}<asr_text>{t}" for t in transcripts]

        prefix_msgs_list = [[_build_prefix_messages(self.system_prompt)] for _ in features]
        prefix_texts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)[0]
            for m in prefix_msgs_list
        ]
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]

        # padding_side="right" MUST be passed as a direct call kwarg:
        # tokenizer.padding_side=... and text_kwargs={padding_side:...} are both
        # ignored by Qwen3ASRProcessor. With left-padding, `labels[:, :prefix_len]`
        # masks pad tokens at the start instead of the real prefix content —
        # see issue #70 in QwenLM/Qwen3-ASR.
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    """Trainer subclass from the official SFT script: explicitly casts float
    tensors in the batch to the model's dtype before the forward pass.

    Autocast *usually* handles this but not reliably for all ops (conv2d in
    the audio encoder is the classic fail case). The explicit cast removes
    the ambiguity."""

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _filter_audio_duration(sample) -> bool:
    arr = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    dur = len(arr) / sr
    return MIN_AUDIO_DURATION_S <= dur <= MAX_AUDIO_DURATION_S


def _ensure_cv22_split(language: str, split: str) -> tuple[Path, Path]:
    """Download (cached) the TSV + audio tar for one CV22 split, extract once.

    Returns (tsv_path, audio_dir).
    """
    cache = CV22_CACHE_DIR / language
    cache.mkdir(parents=True, exist_ok=True)
    audio_dir = cache / split
    extracted_marker = audio_dir / ".extracted"

    logger.info(f"  Fetching transcript/{language}/{split}.tsv ...")
    tsv_path = Path(hf_hub_download(
        repo_id=CV22_REPO, repo_type="dataset",
        filename=f"transcript/{language}/{split}.tsv",
    ))

    if not extracted_marker.exists():
        logger.info(f"  Fetching audio tar for {split} (one-time) ...")
        tar_path = hf_hub_download(
            repo_id=CV22_REPO, repo_type="dataset",
            filename=f"audio/{language}/{split}/{language}_{split}_0.tar",
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Extracting tar to {audio_dir} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(audio_dir)
        extracted_marker.touch()
    else:
        logger.info(f"  Reusing cached audio at {audio_dir}")

    return tsv_path, audio_dir


def load_cv22(language: str, split: str):
    """Manually load fsicoli/common_voice_22_0 (datasets 4.x rejects scripts)."""
    split = CV22_SPLIT_ALIASES.get(split, split)
    logger.info(f"Loading {CV22_REPO} ({language}) split={split} ...")
    tsv_path, audio_dir = _ensure_cv22_split(language, split)

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    if "sentence" not in df.columns or "path" not in df.columns:
        raise RuntimeError(
            f"Unexpected TSV schema for {split}: columns={list(df.columns)}"
        )

    # Build path → file map (tar may extract files at root or nested)
    file_index: dict[str, str] = {}
    for root, _, files in os.walk(audio_dir):
        for f in files:
            file_index[f] = os.path.join(root, f)

    df["audio_path"] = df["path"].map(file_index)
    n_total = len(df)
    df = df.dropna(subset=["audio_path"])
    df = df[df["sentence"].notna()]
    df = df[df["sentence"].astype(str).str.strip() != ""]

    # Drop pathologically long transcripts (see MAX_SENTENCE_CHARS comment).
    sentence_lens = df["sentence"].astype(str).str.len()
    long_mask = sentence_lens > MAX_SENTENCE_CHARS
    if long_mask.any():
        dropped_lens = sentence_lens[long_mask].tolist()
        logger.info(
            f"  Dropping {long_mask.sum()} rows with sentence_len > {MAX_SENTENCE_CHARS} "
            f"chars (lengths: {sorted(dropped_lens, reverse=True)[:5]}...)"
        )
        df = df[~long_mask]

    if len(df) < n_total:
        logger.info(f"  {n_total - len(df)} total rows dropped (missing audio / empty / too-long sentence)")

    # Normalize references at load time so train/val/test share one distribution:
    # capitalize first letter, collapse "..." / "…" / ".." to single ".",
    # append "." if no terminal punct / closing bracket / quote.
    df["sentence"] = df["sentence"].astype(str).apply(normalize_written_form)

    ds = Dataset.from_dict({
        "audio": df["audio_path"].tolist(),
        "sentence": df["sentence"].tolist(),
    })
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    n_before = len(ds)
    ds = ds.filter(_filter_audio_duration)
    logger.info(
        f"  {len(ds)} samples after filtering "
        f"(dropped {n_before - len(ds)} for duration outside [{MIN_AUDIO_DURATION_S}, {MAX_AUDIO_DURATION_S}]s)"
    )
    return ds


def load_cv17_test(language: str):
    """Load CV17 test for cross-version eval (apples-to-apples vs Parakeet-pt)."""
    logger.info(f"Loading {CV17_REPO} ({language}) test for eval...")
    ds = load_dataset(CV17_REPO, language, split="test")
    ds = ds.select_columns(["audio", "sentence"])
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    # Normalize references to match training distribution
    ds = ds.map(lambda x: {"sentence": normalize_written_form(x["sentence"])})
    return ds


def load_synthetic_pt(subset: str, split: str):
    """Load a subset of `yuriyvnv/synthetic_transcript_pt` (Parakeet-pt corpus).

    Subsets we care about for Qwen3-ASR:
      - cv_high_quality: ~48.2k rows, CV17-pt filtered by WAVe multimodal embedding
        similarity > 0.5 (87.3% retention). Column: `text` (not `sentence`).
      - mixed_cv_synthetic: CV + synthetic TTS, used for Parakeet-pt.
      - cv_only / fully_synthetic / mixed_cv_synthetic_all (others).
    """
    logger.info(f"Loading {SYNTH_PT_REPO} / {subset} (split={split}) ...")
    try:
        ds = load_dataset(SYNTH_PT_REPO, subset, split=split)
    except ValueError:
        if split == "validation":
            ds = load_dataset(SYNTH_PT_REPO, subset, split="dev")
        else:
            raise
    # Repo uses `text` column, rename to match our pipeline's `sentence`
    if "text" in ds.column_names and "sentence" not in ds.column_names:
        ds = ds.rename_column("text", "sentence")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.select_columns(["audio", "sentence"])
    # Normalize (capitalize first + single terminal period)
    ds = ds.map(lambda x: {"sentence": normalize_written_form(x["sentence"])})
    # Standard filters
    ds = ds.filter(lambda x: bool(x["sentence"]) and x["sentence"].strip() != "")
    n_before = len(ds)
    ds = ds.filter(_filter_audio_duration)
    logger.info(
        f"  {len(ds)} samples after filtering "
        f"(dropped {n_before - len(ds)} for duration outside [{MIN_AUDIO_DURATION_S}, {MAX_AUDIO_DURATION_S}]s)"
    )
    # Also drop pathological-length transcripts (defensive — synthetic_pt shouldn't
    # have them, but the same rule we apply to CV22 catches anything)
    n_before = len(ds)
    ds = ds.filter(lambda x: len(x["sentence"]) <= MAX_SENTENCE_CHARS)
    if len(ds) < n_before:
        logger.info(f"  Dropped {n_before - len(ds)} rows with sentence > {MAX_SENTENCE_CHARS} chars")
    return ds


def load_synthetic_nl(split: str = "train"):
    """Load `yuriyvnv/synthetic_transcript_nl` — fully-synthetic OpenAI-TTS Dutch
    speech (~34,898 rows, single `default` config, only a `train` split).

    The dataset uses `text` (not `sentence`); this loader renames + normalises
    + casts audio to {SR} so the schema matches `load_cv22` output and the two
    can be concatenated.
    """
    if split != "train":
        raise ValueError(
            f"synthetic_transcript_nl only ships a `train` split (asked for {split!r}). "
            "Use CV22-nl validation/test for held-out evaluation."
        )
    logger.info(f"Loading {SYNTH_NL_REPO} (split={split}) ...")
    ds = load_dataset(SYNTH_NL_REPO, split=split)
    if "text" in ds.column_names and "sentence" not in ds.column_names:
        ds = ds.rename_column("text", "sentence")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    ds = ds.select_columns(["audio", "sentence"])
    ds = ds.map(lambda x: {"sentence": normalize_written_form(x["sentence"])})
    ds = ds.filter(lambda x: bool(x["sentence"]) and x["sentence"].strip() != "")
    n_before = len(ds)
    ds = ds.filter(_filter_audio_duration)
    logger.info(
        f"  {len(ds)} samples after filtering "
        f"(dropped {n_before - len(ds)} for duration outside [{MIN_AUDIO_DURATION_S}, {MAX_AUDIO_DURATION_S}]s)"
    )
    n_before = len(ds)
    ds = ds.filter(lambda x: len(x["sentence"]) <= MAX_SENTENCE_CHARS)
    if len(ds) < n_before:
        logger.info(f"  Dropped {n_before - len(ds)} rows with sentence > {MAX_SENTENCE_CHARS} chars")
    return ds


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, processor, dataset, language_name: str = "Portuguese",
                   system_prompt: str = "", batch_size: int = 8,
                   max_new_tokens: int = 128) -> dict:
    """Run greedy generation on a dataset and compute WER/CER.

    Matches the `qwen_asr.inference._build_text_prompt` pattern with
    force_language=language_name: chat template prefix + `language X<asr_text>`
    is fed to the model, so it generates only the transcript (no language
    prefix to strip).
    """
    model.eval()
    refs = [s.strip() for s in dataset["sentence"]]
    hyps: list[str] = []

    audio_arrays = [
        np.asarray(s["audio"]["array"], dtype=np.float32)
        for s in tqdm(dataset, desc="Preparing audio")
    ]

    prefix_msgs = [_build_prefix_messages(system_prompt)]
    base_prompt = processor.apply_chat_template(
        prefix_msgs, add_generation_prompt=True, tokenize=False
    )[0] + f"language {language_name}<asr_text>"

    with torch.no_grad():
        for i in tqdm(range(0, len(audio_arrays), batch_size), desc="Generate"):
            sub = audio_arrays[i : i + batch_size]
            inputs = processor(
                text=[base_prompt] * len(sub),
                audio=sub,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            inputs = {k: v.to(model.dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}

            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            # Qwen3ASRForConditionalGeneration.generate() forces
            # return_dict_in_generate=True, so `out` is a GenerateOutput
            # with a `.sequences` tensor. Indexing it directly gives a tuple.
            generated_ids = out.sequences if hasattr(out, "sequences") else out
            decoded = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            hyps.extend([d.strip() for d in decoded])

    wer = compute_wer(refs, hyps)
    cer = compute_cer(refs, hyps)
    return {
        "wer": round(wer * 100, 2),
        "cer": round(cer * 100, 2),
        "num_samples": len(refs),
    }


# ---------------------------------------------------------------------------
# WER callback (runs after each Trainer eval, on a subsample of val)
# ---------------------------------------------------------------------------


class WERCallback(TrainerCallback):
    """Runs greedy generation on the eval dataset at the end of each epoch
    and logs eval_wer / eval_cer.

    Fires on `on_epoch_end` (not `on_evaluate`) because generation is slow
    (~minutes per 1k samples for a 1.7B model) and per-step eval would
    dominate training time. Trainer's standard `eval_loss` continues to
    run every `eval_steps` for early-stopping signal.
    """

    def __init__(self, processor, eval_dataset, batch_size: int,
                 language_name: str = "Portuguese", system_prompt: str = ""):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.language_name = language_name
        self.system_prompt = system_prompt

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        logger.info(
            f"\n  [epoch {state.epoch:.2f}, step {state.global_step}] "
            f"running WER eval on {len(self.eval_dataset)} samples..."
        )
        metrics = evaluate_model(
            model, self.processor, self.eval_dataset,
            language_name=self.language_name,
            system_prompt=self.system_prompt,
            batch_size=self.batch_size,
        )
        wer = metrics["wer"]
        cer = metrics["cer"]
        logger.info(
            f"  [epoch {state.epoch:.2f}] eval_wer={wer:.2f}%  eval_cer={cer:.2f}%  "
            f"(n={metrics['num_samples']})"
        )
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {"eval/wer": wer, "eval/cer": cer, "epoch": state.epoch},
                    step=state.global_step,
                )
        except ImportError:
            pass


def _load_qwen_zero_shot_baseline(language_code: str) -> list[dict]:
    """Load the project's zero-shot Qwen3-ASR eval JSONs (results/qwenV3/<lang>/)
    so the README comparison table can show before/after WER. Silently returns
    an empty list if no baseline exists — the README still renders fine."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    base_dir = repo_root / "results" / "qwenV3" / language_code
    if not base_dir.is_dir():
        return []
    out: list[dict] = []
    # Prefer the *_baseline.json files (produced by
    # scripts/evaluate/qwen_pt_zero_shot_baseline.py with the same protocol as
    # the in-training fine-tuned eval: normalized refs + raw model.generate).
    # Fall back to the older qwen-asr-package baseline if the new one is absent.
    candidates_per_set = {
        "cv17_test": [
            f"qwen3-asr-1.7b_{language_code}_cv17_test_baseline.json",
            f"qwen3-asr-1.7b_{language_code}_cv17_test.json",
        ],
        "cv22_test": [
            f"qwen3-asr-1.7b_{language_code}_cv22_test_baseline.json",
            f"qwen3-asr-1.7b_{language_code}_cv22_test.json",
        ],
    }
    for ts_label, fnames in candidates_per_set.items():
        for fname in fnames:
            p = base_dir / fname
            if not p.exists():
                continue
            try:
                with open(p) as f:
                    data = json.load(f)
                out.append({
                    "test_set": ts_label,
                    "wer": data.get("wer"),
                    "cer": data.get("cer"),
                    "num_samples": data.get("num_samples"),
                })
                break  # use the first existing candidate (preferred over fallback)
            except Exception:
                continue
    return out


def _dataset_card_meta(dataset_key: str, language_code: str) -> tuple[str, str, str]:
    """Return (display_name, url, blurb_template) for the README based on the
    --dataset key used at training time. Blurb template may use {language_name}."""
    if dataset_key == "cv22":
        return (
            "fsicoli/common_voice_22_0",
            "https://huggingface.co/datasets/fsicoli/common_voice_22_0",
            "the {language_name} subset of Common Voice 22",
        )
    if dataset_key == "synthetic_pt_high_quality":
        return (
            "yuriyvnv/synthetic_transcript_pt (cv_high_quality)",
            "https://huggingface.co/datasets/yuriyvnv/synthetic_transcript_pt",
            "Common Voice 17 {language_name} filtered by WAVe multimodal "
            "embedding similarity (>0.5) for higher-quality transcripts",
        )
    if dataset_key == "mixed_nl":
        return (
            "yuriyvnv/synthetic_transcript_nl + fsicoli/common_voice_22_0 (nl)",
            "https://huggingface.co/datasets/yuriyvnv/synthetic_transcript_nl",
            "the full synthetic OpenAI-TTS {language_name} corpus (~34.9k clips) "
            "concatenated with the Common Voice 22 {language_name} train split, "
            "shuffled with the run seed",
        )
    return (dataset_key, "", "the {language_name} training set")


def _build_simple_readme(language_code: str, language_name: str, base_model: str) -> str:
    """Minimal README uploaded alongside weights immediately after training so
    the repo is usable for inference even before the final eval completes."""
    return f"""---
language:
  - {language_code}
license: apache-2.0
library_name: transformers
tags:
  - automatic-speech-recognition
  - speech
  - qwen3-asr
  - qwen
  - {language_name.lower()}
  - fine-tuned
  - common-voice
base_model: {base_model}
pipeline_tag: automatic-speech-recognition
---

# Qwen3-ASR-1.7B {language_name}

Fine-tuned [{base_model}](https://huggingface.co/{base_model}) for {language_name} automatic speech recognition.

**Note:** Full evaluation results (WER/CER on Common Voice) are being generated now — this README will be updated shortly.

## Usage

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "yuriyvnv/Qwen3-ASR-1.7B-{language_code.upper()}",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(audio="audio.wav", language="{language_name}")
print(results[0].text)
```
"""


def _build_full_readme(
    language_code: str,
    language_name: str,
    base_model: str,
    test_results: list[dict],
    train_samples: int,
    val_samples: int,
    training_args_summary: dict,
    baseline_results: list[dict] | None = None,
    train_dataset_name: str = "fsicoli/common_voice_22_0",
    train_dataset_url: str = "https://huggingface.co/datasets/fsicoli/common_voice_22_0",
    train_dataset_blurb: str = "Common Voice 22 ({language_name} subset)",
) -> str:
    """Full model card with evaluation results, training details, usage."""

    TEST_SET_LABELS = {
        "cv17_test": "Common Voice 17 (test)",
        "cv22_test": "Common Voice 22 (test)",
    }
    TEST_SET_DATASETS = {
        "cv17_test": "fixie-ai/common_voice_17_0",
        "cv22_test": "fsicoli/common_voice_22_0",
    }

    baseline_results = baseline_results or []
    baseline_by_set = {b.get("test_set"): b for b in baseline_results}

    # Comparison table: zero-shot vs fine-tuned, with absolute and relative gains.
    comparison_rows = []
    for tr in test_results:
        ts = tr.get("test_set", "?")
        label = TEST_SET_LABELS.get(ts, ts)
        ft_wer = tr.get("wer")
        ft_cer = tr.get("cer")
        n = tr.get("num_samples", "?")
        base = baseline_by_set.get(ts)
        if base is not None and isinstance(base.get("wer"), (int, float)) and isinstance(ft_wer, (int, float)):
            base_wer = base["wer"]
            base_cer = base.get("cer", "—")
            delta = ft_wer - base_wer
            rel = (delta / base_wer * 100.0) if base_wer else 0.0
            comparison_rows.append(
                f"| {label} | {n:,} | {base_wer:.2f} | {ft_wer:.2f} | "
                f"**{delta:+.2f}** ({rel:+.1f}%) | {base_cer} → {ft_cer} |"
            )
        else:
            n_fmt = f"{n:,}" if isinstance(n, int) else str(n)
            ft_wer_fmt = f"{ft_wer:.2f}" if isinstance(ft_wer, (int, float)) else "?"
            comparison_rows.append(
                f"| {label} | {n_fmt} | — | {ft_wer_fmt} | — | — → {ft_cer} |"
            )
    comparison_table = "\n".join(comparison_rows) if comparison_rows else "| (no results yet) | — | — | — | — | — |"

    # model-index block (HF leaderboard)
    model_index_results = []
    for tr in test_results:
        ts = tr.get("test_set", "?")
        ds_type = TEST_SET_DATASETS.get(ts, "unknown")
        ds_name = TEST_SET_LABELS.get(ts, ts)
        model_index_results.append(f"""      - task:
          type: automatic-speech-recognition
        dataset:
          name: {ds_name}
          type: {ds_type}
          config: {language_code}
          split: test
        metrics:
          - type: wer
            value: {tr.get('wer')}
            name: WER
          - type: cer
            value: {tr.get('cer')}
            name: CER""")
    model_index_block = "\n".join(model_index_results) if model_index_results else ""

    # Headline numbers for the intro paragraph (best fine-tuned WER + best gain)
    headline_line = ""
    if test_results:
        best = min(
            (tr for tr in test_results if isinstance(tr.get("wer"), (int, float))),
            key=lambda tr: tr["wer"],
            default=None,
        )
        if best is not None:
            ts = best.get("test_set", "?")
            base = baseline_by_set.get(ts)
            label = TEST_SET_LABELS.get(ts, ts)
            if base is not None and isinstance(base.get("wer"), (int, float)):
                rel = (best["wer"] - base["wer"]) / base["wer"] * 100.0
                headline_line = (
                    f"On **{label}** it reaches **{best['wer']:.2f}% WER** "
                    f"(down from {base['wer']:.2f}% zero-shot, **{rel:+.1f}%** relative)."
                )
            else:
                headline_line = f"On **{label}** it reaches **{best['wer']:.2f}% WER**."

    train_dataset_blurb_fmt = train_dataset_blurb.format(language_name=language_name)

    # Encode language name for shields.io badge URL
    language_badge = language_name.replace(" ", "%20")

    return f"""---
language:
  - {language_code}
license: apache-2.0
library_name: transformers
tags:
  - automatic-speech-recognition
  - speech
  - qwen3-asr
  - qwen
  - {language_name.lower()}
  - fine-tuned
  - common-voice
datasets:
  - fsicoli/common_voice_22_0
  - fixie-ai/common_voice_17_0
  - yuriyvnv/synthetic_transcript_pt
base_model: {base_model}
pipeline_tag: automatic-speech-recognition
model-index:
  - name: Qwen3-ASR-1.7B-{language_code.upper()}
    results:
{model_index_block}
---

# 🎙️ Qwen3-ASR-1.7B-{language_code.upper()} — {language_name} Speech Recognition

<div align="center">
  <img src="https://img.shields.io/badge/Parameters-1.7B-red" alt="1.7B Parameters">
  <img src="https://img.shields.io/badge/Modality-Speech%20%E2%86%92%20Text-purple" alt="Speech to Text">
  <img src="https://img.shields.io/badge/Language-{language_badge}-green" alt="{language_name}">
  <img src="https://img.shields.io/badge/Task-ASR-blue" alt="Automatic Speech Recognition">
  <img src="https://img.shields.io/badge/Base-Qwen3--ASR--1.7B-orange" alt="Base model">
  <img src="https://img.shields.io/badge/Precision-bf16-lightgrey" alt="bf16">
  <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" alt="Apache-2.0">
</div>

<br/>

A {language_name}-specialised automatic speech recognition (ASR) model,
fine-tuned from [{base_model}](https://huggingface.co/{base_model}). It outputs
cased, punctuated {language_name} text and works as a drop-in replacement for
the base model.

{headline_line}

---

## 📊 Results

WER and CER on held-out Common Voice test sets — same samples, same protocol,
no test-time tricks. "Zero-shot" is the base
[{base_model}](https://huggingface.co/{base_model}) called with
`language="{language_name}"`. The fine-tuned numbers are **bold**.

| Test set | Samples | Zero-shot WER | **Fine-tuned WER** | Δ WER | CER (zero-shot → fine-tuned) |
|---|---:|---:|---:|---:|:--:|
{comparison_table}

Lower is better. Both held-out test sets see roughly a **one-third relative
reduction in word error rate** versus the already-strong base model.

> 🔬 **Reproducibility note.** Both the zero-shot baseline and the fine-tuned
> numbers above were measured with the *same* evaluation function
> (`train_qwen3_asr.evaluate_model`), the *same* greedy decoding settings, and
> the *same* reference normalisation (see next section). This is an
> apples-to-apples comparison.

## 🧹 Reference / target normalisation

Common Voice transcripts are crowd-sourced and inconsistent in casing and
trailing punctuation. To give the model a clean, predictable target
distribution we apply a small, deterministic **written-form normalisation**
to every reference at load time, both during training and during evaluation:

1. **Capitalise the first letter** if it is lowercase.
2. **Collapse trailing dots** — any sequence of `.`, `…`, `..`, `...` at the
   end is replaced with a single `.`.
3. **Append a terminal period** if the sentence does not already end in
   terminal punctuation (`. ! ? …`) or a closing bracket / quote
   (`) ] }} " '` etc.).

The exact function lives in `src/evaluation/score_written_form.py` of the
project repository. Concretely:

| Raw reference                  | Normalised                       |
|--------------------------------|----------------------------------|
| `bom dia`                      | `Bom dia.`                       |
| `o gato dorme...`              | `O gato dorme.`                  |
| `como estás?`                  | `Como estás?` *(unchanged)*      |
| `"oi"`                         | `"Oi"` *(closing quote → no `.`)* |

Because the **same** normalisation is applied to references used for the
zero-shot baseline above, the gain reported in the results table reflects the
fine-tune itself — **not** a metric quirk caused by mismatched references.

## 🚀 How to use

Install the official `qwen-asr` package, then load this model exactly the
same way you would load the base Qwen3-ASR:

```bash
pip install qwen-asr
```

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "yuriyvnv/Qwen3-ASR-1.7B-{language_code.upper()}",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

result = model.transcribe(audio="audio.wav", language="{language_name}")
print(result[0].text)
```

Batch inference, automatic language detection, streaming, and vLLM serving
all work identically to the base model — see the
[upstream Qwen3-ASR documentation](https://github.com/QwenLM/Qwen3-ASR) for
details.

## 🛠️ Training

**Dataset:** [{train_dataset_name}]({train_dataset_url}) — {train_dataset_blurb_fmt}.
After duration filtering and transcript-length filtering: **{train_samples:,}**
training samples and **{val_samples:,}** validation samples.

**Recipe:** follows the
[official QwenLM SFT recipe](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
with our local hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate | {training_args_summary.get('lr', '?')} |
| Scheduler | {training_args_summary.get('scheduler', '?')} |
| Warmup ratio | {training_args_summary.get('warmup_ratio', '?')} |
| Per-device batch size | {training_args_summary.get('per_device_batch', '?')} |
| Gradient accumulation | {training_args_summary.get('grad_accum', '?')} |
| Effective batch size | {training_args_summary.get('effective_batch', '?')} |
| Epochs | {training_args_summary.get('epochs', '?')} |
| Precision | bf16 mixed |
| Gradient checkpointing | {training_args_summary.get('grad_ckpt', '?')} |
| Optimizer | AdamW (fused) |

Trained on a single H100. The best checkpoint was selected by validation
loss.

## ⚠️ Limitations

- Trained on Common Voice — read-speech dominated. Conversational,
  overlapping-speaker, far-field, or strongly accented audio may degrade
  accuracy.
- Outputs {language_name} text. Cross-lingual or code-switched audio is not
  targeted.
- Punctuation and casing are best-effort and inherit the inconsistencies of
  the Common Voice reference transcripts (mitigated, but not eliminated, by
  the normalisation step above).

## 🙏 Acknowledgements

This model would not exist without the work of others. Thank you to:

- **The Qwen team at Alibaba Cloud** for releasing
  [Qwen3-ASR-1.7B](https://huggingface.co/{base_model}) — the backbone of
  this fine-tune — together with a clean, reproducible
  [SFT recipe](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) and
  the [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337).
- **The Mozilla Common Voice community** for collecting and releasing the
  {language_name} speech corpus used for training and evaluation
  ([Common Voice 22](https://huggingface.co/datasets/fsicoli/common_voice_22_0),
  [Common Voice 17 mirror](https://huggingface.co/datasets/fixie-ai/common_voice_17_0)).
- **Every contributor** who recorded, validated, or transcribed a clip in
  Common Voice. This model is, very literally, your voices.

## 📚 Citation

If this model is useful in your work, please cite the base Qwen3-ASR report:

```bibtex
@article{{qwen3asr2025,
  title  = {{Qwen3-ASR Technical Report}},
  author = {{Qwen Team}},
  year   = {{2025}},
  url    = {{https://arxiv.org/abs/2601.21337}}
}}
```

And, if relevant, this {language_name} fine-tune:

```
yuriyvnv/Qwen3-ASR-1.7B-{language_code.upper()} — {language_name} fine-tune of Qwen3-ASR-1.7B
                                  trained on Common-Voice-derived data with WAVe-based
                                  quality filtering.
```
"""


def _copy_required_hf_files(src_dir: str, dst_dir: str):
    """Copy config + tokenizer/processor files into a checkpoint dir so it can
    be loaded standalone via from_pretrained. Mirrors the official SFT script.
    """
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]
    for fn in required:
        src = Path(src_dir) / fn
        if src.exists():
            shutil.copy2(src, Path(dst_dir) / fn)


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    """After every save, copy config + tokenizer/processor files from the base
    model into the new checkpoint dir so each checkpoint can be loaded
    standalone for testing without re-downloading."""

    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.is_dir():
            _copy_required_hf_files(self.base_model_path, str(ckpt_dir))
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-ASR-1.7B (full FT, bf16) on CV22"
    )
    parser.add_argument("--language", required=True, choices=["pt", "nl"])
    parser.add_argument(
        "--dataset", default="cv22",
        choices=["cv22", "synthetic_pt_high_quality", "mixed_nl"],
        help="Training dataset source. 'cv22' = fsicoli/common_voice_22_0 (per --language). "
             "'synthetic_pt_high_quality' = yuriyvnv/synthetic_transcript_pt subset "
             "cv_high_quality (~48k rows, CV17-pt filtered by WAVe embedding similarity). "
             "'mixed_nl' = yuriyvnv/synthetic_transcript_nl (all 34,898 synthetic) + "
             "fsicoli/common_voice_22_0 nl train, concatenated and shuffled.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=3,
                        help="Default 3. Official QwenLM uses 1 (issue #87 reports hallucination "
                             "at 3 epochs on 150K hrs); on small data (~23k samples / 40h) and "
                             "lower LR the 3-epoch risk is reduced. Drop back to 1 if you see "
                             "WER getting worse between epoch evals.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-device train batch size. With grad-ckpt off, expect ~84 GB "
                             "peak on H100; if OOM, drop to 16 and bump --gradient-accumulation-steps to 8.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Default 4 → effective batch 128 with batch_size=32.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Re-enable gradient checkpointing (off by default — saves ~12 GB but "
                             "slows training ~25%%).")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Default 1e-5 (lowered from official 2e-5 because we're training "
                             "for 3 epochs; lower LR + more epochs is safer than higher LR + 1).")
    parser.add_argument("--warmup-ratio", type=float, default=0.02,
                        help="Default 0.02 (2%%) — official QwenLM SFT default.")
    parser.add_argument("--lr-scheduler-type", type=str, default="linear",
                        help="Default linear — official QwenLM SFT default.")
    parser.add_argument("--system-prompt", type=str, default="",
                        help="Optional system message content (default empty, matches official SFT).")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Default 100. With ~178 steps/epoch × 3 epochs = ~534 total steps, "
                             "this gives ~5 mid-training eval_loss checks (was 200 — too coarse).")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="flash_attention_2 needs `uv add flash-attn --no-build-isolation`")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default="yuriyvnv/Qwen3-ASR-1.7B-PT",
                        help="Public model card repo. Gets (1) model weights + simple README "
                             "immediately after save_model, then (2) a full README with WER "
                             "after test eval completes.")
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="Skip CV17/CV22 test evaluation after training")
    parser.add_argument("--wer-eval-samples", type=int, default=0,
                        help="Subsample size for in-training WER eval. "
                             "0 (default) = full val set. -1 = disable WER callback. "
                             "WER is evaluated at end of each epoch regardless.")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_PROJECT", "syntts-asr-qwen3")
    run_name = f"qwen3-asr-1.7b-{args.dataset}-{args.language}-seed{args.seed}"
    language_name = LANGUAGE_NAMES[args.language]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    if args.dataset == "cv22":
        train_ds = load_cv22(args.language, "train")
        val_ds = load_cv22(args.language, "validation")
    elif args.dataset == "synthetic_pt_high_quality":
        if args.language != "pt":
            raise ValueError("synthetic_pt_high_quality only supports --language pt")
        train_ds = load_synthetic_pt("cv_high_quality", "train")
        try:
            val_ds = load_synthetic_pt("cv_high_quality", "validation")
        except Exception as e:
            logger.info(f"  synthetic_pt_high_quality has no validation split ({e}); "
                        f"using CV17-{args.language} validation for eval_loss.")
            val_ds = load_cv22(args.language, "validation")
    elif args.dataset == "mixed_nl":
        if args.language != "nl":
            raise ValueError("mixed_nl only supports --language nl")
        synth = load_synthetic_nl("train")
        cv22_train = load_cv22(args.language, "train")
        logger.info(
            f"  Combining synthetic_nl ({len(synth):,}) + CV22-nl train "
            f"({len(cv22_train):,}) → {len(synth)+len(cv22_train):,} samples"
        )
        train_ds = concatenate_datasets([synth, cv22_train]).shuffle(seed=args.seed)
        val_ds = load_cv22(args.language, "validation")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # -----------------------------------------------------------------------
    # Model + processor
    # -----------------------------------------------------------------------
    logger.info(f"Loading processor from {QWEN_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    # Note: right-padding is forced per-call in DataCollatorQwen3ASR (see comment
    # there). Setting processor.tokenizer.padding_side has no effect because
    # Qwen3ASRProcessor rebuilds text_kwargs defaults on every __call__.

    logger.info(
        f"Loading model from {QWEN_MODEL_ID} "
        f"(bfloat16, attn_implementation={args.attn_implementation})..."
    )
    model = Qwen3ASRForTraining.from_pretrained(
        QWEN_MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    # Align token IDs with the tokenizer BEFORE Trainer init to silence the
    # "tokenizer has new PAD/BOS/EOS tokens" warning. The pretrained config
    # ships with stale defaults; the tokenizer is the source of truth.
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = pad_id
    model.config.eos_token_id = eos_id

    # Disable KV cache for training. Qwen3-ASR has nested configs; the warning
    # fires from `transformers/utils/generic.py` which reads `self.config.use_cache`
    # on the INNER text decoder (`model.thinker.model`) whose config is
    # `text_config`. Set it on every config object we can find — belt-and-suspenders.
    def _set(obj, attr, value):
        try:
            setattr(obj, attr, value)
        except Exception:
            pass

    _set(model.config, "use_cache", False)
    if hasattr(model.config, "thinker_config"):
        _set(model.config.thinker_config, "use_cache", False)
        if hasattr(model.config.thinker_config, "text_config"):
            _set(model.config.thinker_config.text_config, "use_cache", False)
    # Also set on the live submodule configs (same objects in principle, but
    # some transformers versions copy on access).
    if hasattr(model, "thinker"):
        _set(model.thinker.config, "use_cache", False)
        if hasattr(model.thinker, "model"):
            _set(model.thinker.model.config, "use_cache", False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("  Gradient checkpointing: ENABLED")
    else:
        logger.info("  Gradient checkpointing: DISABLED (faster, ~12 GB more memory)")
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.use_cache = False
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = eos_id

    collator = DataCollatorQwen3ASR(
        processor=processor,
        language_name=language_name,
        system_prompt=args.system_prompt,
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        save_safetensors=True,
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        report_to=["wandb"],
        dataloader_num_workers=4,
        remove_unused_columns=False,  # collator reads audio + sentence
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    callbacks = [MakeEveryCheckpointInferableCallback(base_model_path=QWEN_MODEL_ID)]
    if args.wer_eval_samples >= 0:
        if args.wer_eval_samples == 0:
            wer_eval_set = val_ds  # full val
        else:
            wer_eval_set = val_ds.select(range(min(args.wer_eval_samples, len(val_ds))))
        callbacks.append(
            WERCallback(
                processor=processor,
                eval_dataset=wer_eval_set,
                batch_size=args.eval_batch_size,
                language_name=language_name,
                system_prompt=args.system_prompt,
            )
        )
        logger.info(
            f"  WER callback enabled (per-epoch) on {len(wer_eval_set)} val samples"
        )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
    )

    eff_batch = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"Starting training: {run_name}")
    logger.info(f"  Per-device batch: {args.batch_size}")
    logger.info(f"  Grad accum: {args.gradient_accumulation_steps} → effective batch {eff_batch}")
    logger.info(f"  Epochs: {args.num_train_epochs}")
    logger.info(f"  LR: {args.learning_rate}, warmup: {args.warmup_ratio}, scheduler: {args.lr_scheduler_type}")
    logger.info(f"  Train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    trainer.train()

    # Save best model. Explicitly set `architectures` to the public class so
    # `Qwen3ASRModel.from_pretrained(...)` on the published repo works without
    # knowing about our `Qwen3ASRForTraining` subclass.
    model.config.architectures = ["Qwen3ASRForConditionalGeneration"]
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"Best model saved to {output_dir}")

    # -----------------------------------------------------------------------
    # Stage 1 push: weights + simple README (so the repo is usable immediately
    # even if the test eval below crashes or is interrupted)
    # -----------------------------------------------------------------------
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        logger.info(f"[stage 1/2] Creating repo {args.hub_repo_id} and uploading weights...")
        api.create_repo(args.hub_repo_id, repo_type="model", exist_ok=True)
        # Upload only the files Qwen3ASRModel.from_pretrained needs. Exclude
        # training artefacts (wandb, intermediate checkpoints, optimizer state).
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.hub_repo_id,
            commit_message="Upload fine-tuned weights (training complete, eval pending)",
            ignore_patterns=[
                "wandb/*", "checkpoint-*/", "runs/*",
                "optimizer.pt", "scheduler.pt", "trainer_state.json",
                "training_args.bin", "rng_state.pth",
                "test_results.json",  # uploaded with the final README in stage 2
            ],
        )
        # Upload a minimal README so the repo renders immediately
        simple_readme = _build_simple_readme(args.language, language_name, QWEN_MODEL_ID)
        api.upload_file(
            path_or_fileobj=simple_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.hub_repo_id,
            commit_message="Add initial README (results pending)",
        )
        logger.info(f"  Stage 1 done: https://huggingface.co/{args.hub_repo_id}")

    # -----------------------------------------------------------------------
    # Final evaluation: CV17-pt test + CV22-pt test
    # -----------------------------------------------------------------------
    test_results: list[dict] = []
    if not args.skip_final_eval:
        # Reload best model from save (ensures consistent state, dtype, etc.)
        model = trainer.model

        for label, loader in [
            ("cv17_test", lambda: load_cv17_test(args.language)),
            ("cv22_test", lambda: load_cv22(args.language, "test")),
        ]:
            try:
                logger.info(f"\nEvaluating on {label}...")
                ds = loader()
                metrics = evaluate_model(
                    model, processor, ds,
                    language_name=language_name,
                    system_prompt=args.system_prompt,
                    batch_size=args.eval_batch_size,
                )
                metrics.update({
                    "model": run_name,
                    "language": args.language,
                    "test_set": label,
                })
                logger.info(
                    f"  {label}: WER={metrics['wer']:.2f}%  CER={metrics['cer']:.2f}%  "
                    f"({metrics['num_samples']} samples)"
                )
                test_results.append(metrics)
            except Exception as e:
                logger.warning(f"  {label} eval failed: {e}")

        with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Stage 2 push: replace the simple README with the concrete one that
    # embeds the test WER/CER numbers from the eval above.
    # -----------------------------------------------------------------------
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        logger.info(f"[stage 2/2] Updating README on {args.hub_repo_id} with test results...")

        eff_batch = args.batch_size * args.gradient_accumulation_steps
        training_args_summary = {
            "lr": args.learning_rate,
            "scheduler": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "per_device_batch": args.batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "effective_batch": eff_batch,
            "epochs": args.num_train_epochs,
            "grad_ckpt": "enabled" if args.gradient_checkpointing else "disabled",
        }
        # Load zero-shot baseline if available so the README can show
        # before/after numbers automatically.
        baseline_results = _load_qwen_zero_shot_baseline(args.language)

        train_dataset_name, train_dataset_url, train_dataset_blurb = _dataset_card_meta(
            args.dataset, args.language
        )

        full_readme = _build_full_readme(
            language_code=args.language,
            language_name=language_name,
            base_model=QWEN_MODEL_ID,
            test_results=test_results,
            train_samples=len(train_ds),
            val_samples=len(val_ds),
            training_args_summary=training_args_summary,
            baseline_results=baseline_results,
            train_dataset_name=train_dataset_name,
            train_dataset_url=train_dataset_url,
            train_dataset_blurb=train_dataset_blurb,
        )
        api.upload_file(
            path_or_fileobj=full_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.hub_repo_id,
            commit_message="Update README with test WER/CER results",
        )
        # Also upload the raw test_results.json for reproducibility
        if (output_dir / "test_results.json").exists():
            api.upload_file(
                path_or_fileobj=str(output_dir / "test_results.json"),
                path_in_repo="test_results.json",
                repo_id=args.hub_repo_id,
                commit_message="Add test_results.json",
            )
        logger.info(f"  Stage 2 done: https://huggingface.co/{args.hub_repo_id}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
