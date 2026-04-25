#!/bin/bash
# =============================================================================
# Run the 0.6B Dutch zero-shot baseline THEN the 0.6B NL fine-tune in one
# command.
#
# Why: the fine-tune's stage-2 README push reads
# results/qwenV3/nl/qwen3-asr-0.6b_nl_{cv17,cv22}_test_baseline.json to render
# the apples-to-apples comparison table. Running the baseline first guarantees
# those files exist by the time the README is built — no manual republish.
#
# Steps:
#   1. Skip baseline if both JSONs already exist (idempotent re-runs).
#   2. Otherwise, measure zero-shot Qwen3-ASR-0.6B on CV17-nl + CV22-nl test
#      using the same evaluate_model() function the fine-tuned eval uses.
#   3. Launch the 0.6B NL fine-tune (scripts/train/qwen_nl_0.6b.sh).
#
# Total runtime: ~10 min baseline + ~8 h training on H100 (NL data is ~3×
# bigger than the PT mix).
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/../.."

BASELINE_DIR="results/qwenV3/nl"
BASELINE_CV17="${BASELINE_DIR}/qwen3-asr-0.6b_nl_cv17_test_baseline.json"
BASELINE_CV22="${BASELINE_DIR}/qwen3-asr-0.6b_nl_cv22_test_baseline.json"

if [ -f "$BASELINE_CV17" ] && [ -f "$BASELINE_CV22" ]; then
    echo "=== 0.6B NL zero-shot baseline already exists — skipping ==="
    echo "  $BASELINE_CV17"
    echo "  $BASELINE_CV22"
else
    echo "=== Step 1/2: measuring 0.6B NL zero-shot baseline (~10 min) ==="
    uv run python scripts/evaluate/qwen_pt_zero_shot_baseline.py \
        --base-model Qwen/Qwen3-ASR-0.6B \
        --language nl
    echo "=== Baseline done ==="
fi

echo ""
echo "=== Step 2/2: launching 0.6B NL fine-tune ==="
exec bash scripts/train/qwen_nl_0.6b.sh
