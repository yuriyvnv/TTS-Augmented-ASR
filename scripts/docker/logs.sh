#!/usr/bin/env bash
# Tail the training log written via `tee -a results/training.log` inside
# train.sh. Survives docker exec disconnects. Ctrl+C to stop tailing
# (training keeps running).
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG="${LOG:-results/training.log}"
if [ ! -f "${LOG}" ]; then
    echo "Log not found: ${LOG}"
    echo "Has training started? Run: bash scripts/docker/train.sh"
    exit 1
fi
exec tail -f "${LOG}"
