#!/bin/bash
# =============================================================================
# Fine-tune Qwen3-ASR-1.7B for Portuguese (full FT, bf16, no LoRA).
#
# Default dataset: yuriyvnv/synthetic_transcript_pt / cv_high_quality
# (~48k rows, CV17-pt filtered by WAVe embedding similarity > 0.5).
# Alternative: DATASET="cv22" → fsicoli/common_voice_22_0 pt.
#
# Hyperparameters follow QwenLM's official SFT recipe
# (https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning) with our local
# overrides for multi-epoch training. All train/val/test targets are normalised
# at load time: capitalise first letter, collapse trailing "..."/"…"/".."
# to a single ".", append "." if no terminal punct.
#
# Intended for running inside the Docker container (scripts/docker/up.sh).
# If running natively, you'll need to:
#   - uv sync   (+ av.libs symlinks from CLAUDE.md)
#   - uv run huggingface-cli login
#   - uv run wandb login
#   - switch ATTN to "sdpa" unless you've built flash-attn on the host.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
LANGUAGE="pt"
DATASET="synthetic_pt_high_quality"     # or "cv22"
SEED=42
NUM_EPOCHS=6
BATCH_SIZE=92                           # per-device
GRAD_ACCUM=2                            # eff. batch = 92 * 2 = 184
GRADIENT_CHECKPOINTING=true
EVAL_BATCH_SIZE=12
LEARNING_RATE=2e-5                      # official QwenLM default (was 1e-5 last run)
WARMUP_RATIO=0.02
LR_SCHEDULER="linear"
EVAL_STEPS=100
SAVE_STEPS=100
WER_EVAL_SAMPLES=0                      # 0=full val per epoch, N>0=subsample, -1=disable
ATTN="flash_attention_2"                # Docker image ships flash-attn; set "sdpa" if not available

# Reduce GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU
export CUDA_VISIBLE_DEVICES=0

# Hub — public model card repo. Overwrites on each run; HF keeps commit history.
PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/Qwen3-ASR-1.7B-PT"

# Output directory — name reflects the dataset so runs on different datasets
# don't overwrite each other.
OUTPUT_DIR="./results/qwen3_finetune_${LANGUAGE}/${DATASET}_s${SEED}"

# FFmpeg shared libs from PyAV (only needed for native host runs; inside
# Docker ffmpeg is installed system-wide and this line is a harmless no-op).
if [ -d ".venv/lib/python3.12/site-packages/av.libs" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
fi

# ---------------------------------------------------------------------------
# Verify logins
# ---------------------------------------------------------------------------
echo "=== Checking HuggingFace login ==="
if ! uv run hf auth whoami > /dev/null 2>&1; then
    echo "ERROR: Not logged in to HuggingFace. Run: uv run huggingface-cli login"
    exit 1
fi

echo "=== Checking WandB login ==="
if ! uv run python -c "import wandb; assert wandb.api.api_key" > /dev/null 2>&1; then
    echo "ERROR: WandB not configured. Run: uv run wandb login"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
PUSH_FLAG=""
if [ "$PUSH_TO_HUB" = true ]; then
    PUSH_FLAG="--push-to-hub --hub-repo-id $HUB_REPO"
fi

GRAD_CKPT_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    GRAD_CKPT_FLAG="--gradient-checkpointing"
fi

echo "=== Starting training ==="
echo "  Language:        $LANGUAGE"
echo "  Dataset:         $DATASET"
echo "  Output:          $OUTPUT_DIR"
echo "  Effective batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LR / epochs:     $LEARNING_RATE / $NUM_EPOCHS"
echo "  Attention impl:  $ATTN"
echo "  Push to hub:     $HUB_REPO"

uv run python -m src.training.train_qwen3_asr \
    --language "$LANGUAGE" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --num-train-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --lr-scheduler-type "$LR_SCHEDULER" \
    --eval-steps "$EVAL_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --wer-eval-samples "$WER_EVAL_SAMPLES" \
    --attn-implementation "$ATTN" \
    $GRAD_CKPT_FLAG \
    $PUSH_FLAG

echo "=== Done ==="
echo "Test WER/CER saved to $OUTPUT_DIR/test_results.json"
