#!/usr/bin/env bash
# Stop and remove the training container. Does NOT delete the image or
# any mounted caches (HF cache, syntts cache, results). Safe to run
# after training completes.
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-qwen-training}"

if ! docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q .; then
    echo "No container named '${CONTAINER_NAME}' found."
    exit 0
fi

echo "=== Stopping and removing '${CONTAINER_NAME}' ==="
docker stop "${CONTAINER_NAME}" >/dev/null
docker rm "${CONTAINER_NAME}" >/dev/null
echo "=== Done ==="
