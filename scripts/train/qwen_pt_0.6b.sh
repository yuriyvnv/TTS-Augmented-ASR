#!/bin/bash
# =============================================================================
# Fine-tune Qwen3-ASR-0.6B for Portuguese — v1 phase.
#
# Mirrors the v1 1.7B PT recipe: synthetic_pt_high_quality (WAVe-filtered
# CV17-pt) as training data, validate on CV22-pt validation during training,
# report final WER/CER on CV17-pt + CV22-pt test sets.
#
# Goal: validate that the same training recipe scales down to the 0.6B
# variant before committing to the full mixed_pt_full v2 retrain.
#
# Hyperparameters: identical to the v1 1.7B PT recipe
# (scripts/train/qwen_pt.sh) — same LR, scheduler, warmup, epochs, batches:
#
#   per-device train batch:   92
#   grad-accum:               2   (effective batch = 184)
#   eval batch:               12
#   gradient checkpointing:   ENABLED
#
# Reverted from earlier batch-size bumps (256, 128, 64): the audio encoder
# activations + variable-length 30 s clips made every "smaller" config OOM in
# different ways. Mirroring the proven 1.7B config keeps the recipe consistent
# and gives an apples-to-apples comparison to the published 1.7B model.
# =============================================================================

set -euo pipefail

LANGUAGE="pt"
DATASET="synthetic_pt_high_quality"
SEED=42
BASE_MODEL="Qwen/Qwen3-ASR-0.6B"
NUM_EPOCHS=6
BATCH_SIZE=92                           # mirrors v1 1.7B PT recipe (qwen_pt.sh)
GRAD_ACCUM=2                            # eff. batch = 184 (same as v1 1.7B)
GRADIENT_CHECKPOINTING=true
EVAL_BATCH_SIZE=12                      # mirrors v1 1.7B PT recipe
LEARNING_RATE=2e-5
WARMUP_RATIO=0.02
LR_SCHEDULER="linear"
EVAL_STEPS=100
SAVE_STEPS=100
WER_EVAL_SAMPLES=0
ATTN="flash_attention_2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/Qwen3-ASR-0.6B-PT"

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

echo "=== Starting training (0.6B v1) ==="
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
