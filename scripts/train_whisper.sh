#!/bin/bash
# =============================================================================
# Fine-tune Whisper-large-v3 with the settings used in the paper.
#
# Usage:
#   ./scripts/train_whisper.sh                      # all 8 language/config pairs
#   ./scripts/train_whisper.sh et cv_synth_all_et   # one pair
#
# Optional: HUB_REPO=<user>/<repo> ./scripts/train_whisper.sh ...
#   uploads each run folder to that HuggingFace model repo.
#
# Prerequisites: `hf auth login` (dataset download, optional push), `wandb login`.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Settings (paper configuration)
# ---------------------------------------------------------------------------
SEED=42
BATCH_SIZE=64
GRADIENT_ACCUM=2                       # effective batch = 64 * 2 = 128
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-5
WARMUP_RATIO=0.10
NUM_EPOCHS=5
EVAL_STEPS=25
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
# Train loop
# ---------------------------------------------------------------------------
TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    read -r LANGUAGE CONFIG <<< "${EXPERIMENT}"
    CURRENT=$((CURRENT + 1))

    # Folder name = run name, so evaluate.py labels results the way significance.py expects
    OUTPUT_DIR="./results/whisper_finetune_${LANGUAGE}/whisper-large-v3-${CONFIG}-seed${SEED}"

    echo ""
    echo "============================================================"
    echo " [${CURRENT}/${TOTAL}] whisper-large-v3"
    echo " Language:       ${LANGUAGE}"
    echo " Config:         ${CONFIG}"
    echo " Seed:           ${SEED}"
    echo " Batch size:     ${BATCH_SIZE} x ${GRADIENT_ACCUM} accum = $((BATCH_SIZE * GRADIENT_ACCUM)) effective / ${EVAL_BATCH_SIZE} (eval)"
    echo " LR:             ${LEARNING_RATE}"
    echo " Warmup ratio:   ${WARMUP_RATIO}"
    echo " Epochs:         ${NUM_EPOCHS}"
    echo " Eval steps:     ${EVAL_STEPS}"
    echo " Output:         ${OUTPUT_DIR}"
    echo " Push to Hub:    ${HUB_REPO:-no}"
    echo "============================================================"
    echo ""

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
        ${PUSH_FLAGS[@]+"${PUSH_FLAGS[@]}"}

    # Best model is saved at the folder root; drop intermediate checkpoints to free disk
    echo "Cleaning up checkpoints in ${OUTPUT_DIR}..."
    rm -rf "${OUTPUT_DIR}"/checkpoint-*
    rm -rf "${OUTPUT_DIR}"/runs

    echo ""
    echo "=== [${CURRENT}/${TOTAL}] ${CONFIG} done ==="
done
