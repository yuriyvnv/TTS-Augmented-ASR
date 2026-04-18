#!/bin/bash
# =============================================================================
# Fine-tune Parakeet-TDT-0.6B-v3
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
LANGUAGE="sl"                          # et or sl
CONFIG="cv_synth_no_morph_sl"               # see dataset configs in training README
SEED=42
BATCH_SIZE=32
LEARNING_RATE=5e-5                     # peak LR for CosineAnnealing
WARMUP_RATIO=0.10                      # 10% of total steps
MAX_EPOCHS=100
EARLY_STOPPING_PATIENCE=10

# Hub — uploads the whole results folder
PUSH_TO_HUB=true
HUB_REPO="yuriyvnv/experiments_parakeet"

# Output directory
OUTPUT_DIR="./results/parakeet_finetune_${LANGUAGE}/${CONFIG}_s${SEED}"

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
echo " Language:       ${LANGUAGE}"
echo " Config:         ${CONFIG}"
echo " Seed:           ${SEED}"
echo " Batch size:     ${BATCH_SIZE}"
echo " LR:             ${LEARNING_RATE}"
echo " Warmup ratio:   ${WARMUP_RATIO}"
echo " Max epochs:     ${MAX_EPOCHS}"
echo " Early stopping: ${EARLY_STOPPING_PATIENCE} epochs patience"
echo " Output:         ${OUTPUT_DIR}"
echo " Push to Hub:    ${PUSH_TO_HUB} -> ${HUB_REPO}"
echo "============================================================"
echo ""

PUSH_FLAGS=""
if [[ "${PUSH_TO_HUB}" == "true" ]]; then
    PUSH_FLAGS="--push-to-hub --hub-repo-id ${HUB_REPO}"
fi

uv run python -m src.training.train_parakeet \
    --language "${LANGUAGE}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --max-epochs "${MAX_EPOCHS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    ${PUSH_FLAGS}

echo ""
echo "=== Training complete ==="
