#!/usr/bin/env bash
# Start a persistent training container. Stays alive via `sleep infinity`
# so you can `docker exec` into it, start training in tmux, detach, and
# come back hours later. Training keeps running as long as the container
# is up, regardless of whether your SSH session is connected.
set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE="${IMAGE:-syntts-asr:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-training}"
# Explicit NVIDIA-only device. `--gpus all` triggers Docker's multi-vendor
# CDI auto-discovery on this host and fails with "AMD CDI spec not found"
# (the daemon is configured to scan AMD too but no AMD spec exists). The
# `nvidia.com/gpu=0` form goes straight to NVIDIA's CDI registration.
GPU_DEVICE="${GPU_DEVICE:-nvidia.com/gpu=0}"

# If container already running, just print its status
if docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q .; then
    echo "=== Container '${CONTAINER_NAME}' is already running ==="
    docker ps --filter "name=^${CONTAINER_NAME}$" --format "  {{.Names}}  {{.Status}}  {{.Image}}"
    echo ""
    echo "Enter it with:  bash scripts/docker/shell.sh"
    exit 0
fi

# If container exists but stopped, remove it first
if docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q .; then
    echo "=== Removing stopped container '${CONTAINER_NAME}' ==="
    docker rm "${CONTAINER_NAME}" >/dev/null
fi

# HuggingFace + dataset caches persist on host across container lifetimes
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
SYNTTS_CACHE="$HOME/.cache/syntts_asr"
mkdir -p "${HF_CACHE}" "${SYNTTS_CACHE}"

# .env file must exist for secrets (HF_API_KEY, WANDB_API_KEY, OPENAI_API_KEY)
if [ ! -f .env ]; then
    echo "ERROR: .env not found at $(pwd)/.env — copy .env.example and fill in keys."
    exit 1
fi

echo "=== Starting container '${CONTAINER_NAME}' (image ${IMAGE}, device=${GPU_DEVICE}) ==="
docker run \
    --detach \
    --name "${CONTAINER_NAME}" \
    --device "${GPU_DEVICE}" \
    --shm-size=16g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ulimit nofile=65535:65535 \
    --restart unless-stopped \
    --env-file .env \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --volume "$(pwd)/src:/workspace/src" \
    --volume "$(pwd)/scripts:/workspace/scripts" \
    --volume "$(pwd)/prompts:/workspace/prompts" \
    --volume "$(pwd)/results:/workspace/results" \
    --volume "$(pwd)/pyproject.toml:/workspace/pyproject.toml:ro" \
    --volume "$(pwd)/uv.lock:/workspace/uv.lock:ro" \
    --volume "$(pwd)/README.md:/workspace/README.md:ro" \
    --volume "${HF_CACHE}:/root/.cache/huggingface" \
    --volume "${SYNTTS_CACHE}:/root/.cache/syntts_asr" \
    --workdir /workspace \
    "${IMAGE}" \
    sleep infinity

echo "=== Up ==="
docker ps --filter "name=^${CONTAINER_NAME}$" --format "  {{.Names}}  {{.Status}}"
echo ""
echo "Next:"
echo "  bash scripts/docker/shell.sh         # enter container"
echo "  bash scripts/docker/train.sh         # start training in tmux"
echo "  bash scripts/docker/attach.sh        # re-attach to running training"
echo "  bash scripts/docker/logs.sh          # tail training log"
echo "  bash scripts/docker/down.sh          # stop and remove"
