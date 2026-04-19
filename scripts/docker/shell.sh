#!/usr/bin/env bash
# Open an interactive bash shell inside the running container.
# If you Ctrl+D / exit this shell, the container keeps running.
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-qwen-training}"
exec docker exec -it "${CONTAINER_NAME}" bash
