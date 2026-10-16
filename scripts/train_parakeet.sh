#!/bin/bash
# =============================================================================
# Fine-tune Parakeet-TDT-0.6B-v3 with the settings used in the paper.
#
# Usage:
#   ./scripts/train_parakeet.sh                      # all 8 language/config pairs
#   ./scripts/train_parakeet.sh et cv_synth_all_et   # one pair
#
# Optional: HUB_REPO=<user>/<repo> ./scripts/train_parakeet.sh ...
#   uploads each run folder to that HuggingFace model repo.
#
# Prerequisites: `hf auth login` (dataset download, optional push), `wandb login`.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Settings (paper configuration)
# ---------------------------------------------------------------------------
SEED=42
BATCH_SIZE=32
LEARNING_RATE=5e-5                     # peak LR for CosineAnnealing
WARMUP_RATIO=0.10                      # 10% of total steps
MAX_EPOCHS=100
EARLY_STOPPING_PATIENCE=10
HUB_REPO="${HUB_REPO:-}"               # empty = no upload

if [[ $# -eq 2 ]]; then
    EXPERIMENTS=("$1 $2")
elif [[ $# -eq 0 ]]; then
    EXPERIMENTS=(
        "et cv_only_et" "et cv_synth_all_et" "et cv_synth_no_morph_et" "et cv_synth_unfiltered_et"
        "sl cv_only_sl" "sl cv_synth_all_sl" "sl cv_synth_no_morph_sl" "sl cv_synth_unfiltered_sl"
    )
else
    echo "Usage: $0 [LANGUAGE CONFIG]" >&2
    exit 1
fi

PUSH_FLAGS=()
if [[ -n "${HUB_REPO}" ]]; then
    PUSH_FLAGS=(--push-to-hub --hub-repo-id "${HUB_REPO}")
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    read -r LANGUAGE CONFIG <<< "${EXPERIMENT}"
    CURRENT=$((CURRENT + 1))

    OUTPUT_DIR="./results/parakeet_finetune_${LANGUAGE}/${CONFIG}_s${SEED}"

    echo ""
    echo "============================================================"
    echo " [${CURRENT}/${TOTAL}] parakeet-tdt-0.6b-v3"
    echo " Language:       ${LANGUAGE}"
    echo " Config:         ${CONFIG}"
    echo " Seed:           ${SEED}"
    echo " Batch size:     ${BATCH_SIZE}"
    echo " LR:             ${LEARNING_RATE}"
    echo " Warmup ratio:   ${WARMUP_RATIO}"
    echo " Max epochs:     ${MAX_EPOCHS}"
    echo " Early stopping: ${EARLY_STOPPING_PATIENCE} epochs patience"
    echo " Output:         ${OUTPUT_DIR}"
    echo " Push to Hub:    ${HUB_REPO:-no}"
    echo "============================================================"
    echo ""

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
        ${PUSH_FLAGS[@]+"${PUSH_FLAGS[@]}"}

    echo ""
    echo "=== [${CURRENT}/${TOTAL}] ${CONFIG} done: ${OUTPUT_DIR}/parakeet-tdt-${CONFIG}-seed${SEED}.nemo ==="
done
