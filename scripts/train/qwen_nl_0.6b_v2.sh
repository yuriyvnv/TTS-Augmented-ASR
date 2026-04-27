#!/bin/bash
# =============================================================================
# Fine-tune Qwen3-ASR-0.6B for Dutch — v2 run.
#
# Mirrors the 0.6B PT v2 recipe (scripts/train/qwen_pt_0.6b_v2.sh):
#   - Training set = CV22-nl train + CV22-nl validation + synthetic_transcript_nl
#                    (full 34.9k clips), concatenated and shuffled
#                    (--dataset mixed_nl_full)
#   - In-training eval = CV22-nl test (no separate validation split left).
#     Best checkpoint by eval_loss; over a small number of epochs the bias is
#     minimal. CV17-nl test stays fully held out.
#   - 3 epochs (vs 5 on the 0.6B v1) — v1 reached 9.06%/8.95% WER on
#     CV17/CV22-nl test; v2 trades held-out validation for more training data.
#
# Pushes to the SAME repo as v1 (yuriyvnv/Qwen3-ASR-0.6B-NL). HF preserves
# commit history; the v2 README replaces v1's via --update-note.
# =============================================================================

set -euo pipefail

LANGUAGE="nl"
DATASET="mixed_nl_full"
SEED=42
BASE_MODEL="Qwen/Qwen3-ASR-0.6B"
NUM_EPOCHS=3
BATCH_SIZE=92                           # mirrors v1 1.7B + 0.6B recipes
GRAD_ACCUM=2                            # eff. batch = 184
GRADIENT_CHECKPOINTING=true
EVAL_BATCH_SIZE=12
LEARNING_RATE=2e-5
WARMUP_RATIO=0.02
LR_SCHEDULER="linear"
EVAL_STEPS=200                          # NL train set is ~3× PT, space evals out
SAVE_STEPS=200
WER_EVAL_SAMPLES=0                      # 0 = full val (= CV22-nl test) per epoch
ATTN="flash_attention_2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/Qwen3-ASR-0.6B-NL"

UPDATE_NOTE="v2 release. The v1 run (mixed_nl: synthetic_nl + CV22-nl train, 5 epochs) reached 9.06% WER on CV17-nl test and 8.95% WER on CV22-nl test, a -27%/-28% relative reduction vs the 0.6B zero-shot. For this final release we fold CV22-nl train + validation together with the full synthetic_transcript_nl corpus (~34.9k clips) into one training set to maximise the data the model sees, train for just 3 epochs, and validate only on the held-out CV17-nl and CV22-nl test sets. The goal is the strongest possible 0.6B Dutch model for production use, not another methodological ablation."

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

echo "=== Starting training (0.6B NL v2) ==="
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
    --update-note "$UPDATE_NOTE" \
    $GRAD_CKPT_FLAG \
    $PUSH_FLAG

echo "=== Done ==="
echo "Test WER/CER saved to $OUTPUT_DIR/test_results.json"
