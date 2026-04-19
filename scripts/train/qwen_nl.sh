#!/bin/bash
# =============================================================================
# Fine-tune Qwen3-ASR-1.7B for Dutch (full FT, bf16, no LoRA).
#
# Dataset: yuriyvnv/synthetic_transcript_nl (all 34,898 synthetic OpenAI-TTS
# clips) + fsicoli/common_voice_22_0 (nl train), concatenated and shuffled.
# Validation = CV22-nl validation. Test eval = CV17-nl test + CV22-nl test.
#
# All train / val / test references go through `normalize_written_form`:
#   capitalise first letter, collapse trailing "..."/"…"/".." → ".",
#   append "." if no terminal punct/closing bracket.
#
# Hyperparameters: same as the Portuguese run, except 5 epochs (the combined
# train set is ~3× larger than the PT one, so 5 instead of 6 keeps the
# total step count manageable).
#
# Intended for running inside the Docker container (scripts/docker/up.sh).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
LANGUAGE="nl"
DATASET="mixed_nl"                      # synthetic_nl + CV22-nl train
SEED=42
NUM_EPOCHS=5
BATCH_SIZE=92                           # per-device
GRAD_ACCUM=2                            # eff. batch = 92 * 2 = 184
GRADIENT_CHECKPOINTING=true
EVAL_BATCH_SIZE=12
LEARNING_RATE=2e-5                      # matches PT run
WARMUP_RATIO=0.02
LR_SCHEDULER="linear"
EVAL_STEPS=200                          # ~3× the data → space evals out a bit more
SAVE_STEPS=200
WER_EVAL_SAMPLES=0                      # 0=full val per epoch
ATTN="flash_attention_2"

# Reduce GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU
export CUDA_VISIBLE_DEVICES=0

# Hub — public model card repo. Overwrites on each run; HF keeps commit history.
PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/Qwen3-ASR-1.7B-NL"

# Output directory — name reflects the dataset so runs on different datasets
# don't overwrite each other.
OUTPUT_DIR="./results/qwen3_finetune_${LANGUAGE}/${DATASET}_s${SEED}"

# FFmpeg shared libs from PyAV (only needed for native host runs).
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
