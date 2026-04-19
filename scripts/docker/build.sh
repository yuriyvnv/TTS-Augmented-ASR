#!/usr/bin/env bash
# Build the syntts_asr training image. First build takes ~20–30 min
# (most time is flash-attn compilation). Subsequent builds are layer-cached
# and finish in seconds unless pyproject.toml / uv.lock change.
set -euo pipefail

cd "$(dirname "$0")/../.."

IMAGE_NAME="${IMAGE_NAME:-syntts-asr}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=== Building ${IMAGE_NAME}:${IMAGE_TAG} ==="
echo "    (first build ~20–30 min; flash-attn compilation is the slow part)"

docker build \
  --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
  --progress=plain \
  .

echo "=== Done. Image: ${IMAGE_NAME}:${IMAGE_TAG} ==="
docker images "${IMAGE_NAME}" --format "  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}"
