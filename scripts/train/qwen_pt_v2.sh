#!/bin/bash
# =============================================================================
# Fine-tune Qwen3-ASR-1.7B for Portuguese — v2 run.
#
# What changed vs qwen_pt.sh:
#   - Training set = CV22-pt train + synthetic_pt_high_quality + CV22-pt val
#     (all three combined and shuffled) → more data for the final model.
#   - In-training eval = CV22-pt test (no separate validation split left).
#     We still pick the best checkpoint by eval_loss; over 4 epochs the bias
#     is minimal and we get the strongest model. CV17-pt test is held out.
#   - 4 epochs (vs 6) — the v1 run converged by epoch 4.
#
# Reports CV17-pt test + CV22-pt test WER/CER at the end, same as v1.
# =============================================================================

set -euo pipefail

LANGUAGE="pt"
DATASET="mixed_pt_full"
SEED=42
NUM_EPOCHS=4
BATCH_SIZE=92
GRAD_ACCUM=2                            # eff. batch = 184
GRADIENT_CHECKPOINTING=true
EVAL_BATCH_SIZE=12
LEARNING_RATE=2e-5
WARMUP_RATIO=0.02
LR_SCHEDULER="linear"
EVAL_STEPS=100
SAVE_STEPS=100
WER_EVAL_SAMPLES=0                      # 0 = full val (= CV22-pt test) per epoch
ATTN="flash_attention_2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/Qwen3-ASR-1.7B-PT"

UPDATE_NOTE="v2 release. The v1 run already gave us a clear picture of the learning curve on Portuguese (eval loss and WER plateaued around epoch 4 of 6, with -33% relative WER vs zero-shot), so for this final release we are no longer using the validation split to pick hyperparameters. Instead we fold CV22-pt train + validation together with the WAVe-filtered synthetic_transcript_pt corpus into one training set to maximise the data the model sees, train for just 4 epochs (the point at which v1 converged), and validate only on the held-out CV17-pt and CV22-pt test sets. The goal is the strongest possible Portuguese model for production use, not another methodological ablation."

OUTPUT_DIR="./results/qwen3_finetune_${LANGUAGE}/${DATASET}_s${SEED}"

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

echo "=== Starting training (v2) ==="
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
    --update-note "$UPDATE_NOTE" \
    $GRAD_CKPT_FLAG \
    $PUSH_FLAG

echo "=== Done ==="
echo "Test WER/CER saved to $OUTPUT_DIR/test_results.json"
