#!/usr/bin/env bash
# Reattach to the running tmux training session. Detach again with Ctrl+b d.
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-qwen-training}"
SESSION="${SESSION:-train}"
exec docker exec -it "${CONTAINER_NAME}" tmux attach -t "${SESSION}"
