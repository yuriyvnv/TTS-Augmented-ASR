#!/bin/bash
# =============================================================================
# Fine-tune Qwen3-ASR-0.6B for Dutch (full FT, bf16, no LoRA).
#
# Mirrors the v1 0.6B PT recipe (scripts/train/qwen_pt_0.6b.sh) which, in
# turn, mirrors the 1.7B NL recipe (scripts/train/qwen_nl.sh). Same LR /
# scheduler / warmup / per-device batch / grad accum / grad-ckpt / epochs.
# Only --base-model and --hub-repo-id change.
#
# Dataset: yuriyvnv/synthetic_transcript_nl (all 34,898 synthetic OpenAI-TTS
# clips) + fsicoli/common_voice_22_0 (nl train), concatenated and shuffled.
# Validation = CV22-nl validation. Test eval = CV17-nl test + CV22-nl test.
#
# Hyperparameters (identical to 1.7B NL run):
#   per-device train batch:   92
#   grad-accum:               2   (effective batch = 184)
#   eval batch:               12
#   gradient checkpointing:   ENABLED
#   epochs:                   5  (~3× more data than PT, so 5 instead of 6)
# =============================================================================

set -euo pipefail

LANGUAGE="nl"
DATASET="mixed_nl"
SEED=42
BASE_MODEL="Qwen/Qwen3-ASR-0.6B"
NUM_EPOCHS=5
BATCH_SIZE=92                           # mirrors v1 1.7B NL recipe (qwen_nl.sh)
GRAD_ACCUM=2                            # eff. batch = 184
GRADIENT_CHECKPOINTING=true
EVAL_BATCH_SIZE=12
LEARNING_RATE=2e-5
WARMUP_RATIO=0.02
LR_SCHEDULER="linear"
EVAL_STEPS=200
SAVE_STEPS=200
WER_EVAL_SAMPLES=0
ATTN="flash_attention_2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/Qwen3-ASR-0.6B-NL"

OUTPUT_DIR="./results/qwen3_finetune_${LANGUAGE}/0.6b_${DATASET}_s${SEED}"

if [ -d ".venv/lib/python3.12/site-packages/av.libs" ]; then
    export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
fi

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

PUSH_FLAG=""
if [ "$PUSH_TO_HUB" = true ]; then
    PUSH_FLAG="--push-to-hub --hub-repo-id $HUB_REPO"
fi

GRAD_CKPT_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    GRAD_CKPT_FLAG="--gradient-checkpointing"
fi

echo "=== Starting training (0.6B NL v1) ==="
echo "  Base model:      $BASE_MODEL"
echo "  Language:        $LANGUAGE"
echo "  Dataset:         $DATASET"
echo "  Output:          $OUTPUT_DIR"
echo "  Effective batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  LR / epochs:     $LEARNING_RATE / $NUM_EPOCHS"
echo "  Attention impl:  $ATTN"
echo "  Push to hub:     $HUB_REPO"

uv run python -m src.training.train_qwen3_asr \
    --base-model "$BASE_MODEL" \
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
