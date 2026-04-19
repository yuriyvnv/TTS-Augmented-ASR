#!/usr/bin/env bash
# Kick off training inside a tmux session named 'train' on the running container.
# The tmux session survives you disconnecting — training keeps running.
# Reconnect any time with: bash scripts/docker/attach.sh
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-qwen-training}"
SESSION="${SESSION:-train}"
# Which language script to launch — pass as $1 (e.g. "nl") or set TRAIN_SCRIPT.
# Default: qwen_pt.sh (back-compat with prior runs).
LANG_KEY="${1:-${LANG_KEY:-pt}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/qwen_${LANG_KEY}.sh}"

# Verify container is up
if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q .; then
    echo "ERROR: container '${CONTAINER_NAME}' is not running. Run: bash scripts/docker/up.sh"
    exit 1
fi

# If a tmux session already exists, don't clobber it — attach instead
if docker exec "${CONTAINER_NAME}" tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists — attaching. Detach with Ctrl+b d."
    exec docker exec -it "${CONTAINER_NAME}" tmux attach -t "${SESSION}"
fi

echo "=== Starting training in tmux session '${SESSION}' ==="
echo "    Script:      ${TRAIN_SCRIPT}"
echo "    Detach with: Ctrl+b then d   (training keeps running)"
echo "    Reattach:    bash scripts/docker/attach.sh"
echo ""

# -d creates detached; we then attach so the user sees output from step 0.
# `uv run bash` ensures the script inherits the project's uv-managed env
# (the inner `uv run python -m ...` lines pick up /opt/venv consistently).
docker exec "${CONTAINER_NAME}" tmux new-session -d -s "${SESSION}" \
    "cd /workspace && uv run bash ${TRAIN_SCRIPT} 2>&1 | tee -a results/training_${LANG_KEY}.log; exec bash"

exec docker exec -it "${CONTAINER_NAME}" tmux attach -t "${SESSION}"
