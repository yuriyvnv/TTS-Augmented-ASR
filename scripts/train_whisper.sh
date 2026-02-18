#!/bin/bash
# =============================================================================
# Fine-tune Whisper-large-v3
#
# Prerequisites:
#   1. huggingface-cli login
#   2. wandb login
#
# All settings are defined below — edit before running.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Settings — edit these
# ---------------------------------------------------------------------------
LANGUAGE="et"                          # et or sl
CONFIG="cv_only_et"                    # cv_only_et, cv_synth_all_et, cv_synth_no_morph_et, etc.
SEED=42
BATCH_SIZE=16
GRADIENT_ACCUM=16                      # effective batch = 16 * 16 = 256
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-5
WARMUP_RATIO=0.10
NUM_EPOCHS=5
EVAL_STEPS=50

# Hub — uploads the whole results folder
PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/experiments_parakeet"

# Output directory
OUTPUT_DIR="./results/whisper_finetune_${LANGUAGE}/${CONFIG}_s${SEED}"

# ---------------------------------------------------------------------------
# Verify logins
# ---------------------------------------------------------------------------
echo "=== Checking HuggingFace login ==="
if ! hf auth whoami > /dev/null 2>&1; then
    echo "ERROR: Not logged in to HuggingFace. Run: huggingface-cli login"
    exit 1
fi
echo "  Logged in as: $(hf auth whoami 2>/dev/null | head -1)"

echo "=== Checking WandB login ==="
if ! python -c "import wandb; wandb.api.api_key" > /dev/null 2>&1; then
    echo "WARNING: WandB may not be configured. Run: wandb login"
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Model:          whisper-large-v3"
echo " Language:       ${LANGUAGE}"
echo " Config:         ${CONFIG}"
echo " Seed:           ${SEED}"
echo " Batch size:     ${BATCH_SIZE} x ${GRADIENT_ACCUM} accum = $((BATCH_SIZE * GRADIENT_ACCUM)) effective / ${EVAL_BATCH_SIZE} (eval)"
echo " LR:             ${LEARNING_RATE}"
echo " Warmup ratio:   ${WARMUP_RATIO}"
echo " Epochs:         ${NUM_EPOCHS}"
echo " Eval steps:     ${EVAL_STEPS}"
echo " Output:         ${OUTPUT_DIR}"
echo " Push to Hub:    ${PUSH_TO_HUB} -> ${HUB_REPO}"
echo "============================================================"
echo ""

PUSH_FLAGS=""
if [[ "${PUSH_TO_HUB}" == "true" ]]; then
    PUSH_FLAGS="--push-to-hub --hub-repo-id ${HUB_REPO}"
fi

uv run python -m src.training.train_whisper \
    --language "${LANGUAGE}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUM}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --num-train-epochs "${NUM_EPOCHS}" \
    --eval-steps "${EVAL_STEPS}" \
    ${PUSH_FLAGS}

echo ""
echo "=== Training complete ==="

# Stop the instance after training + push
echo "=== Stopping instance in 30 seconds... ==="
sleep 30
sudo shutdown -h now
