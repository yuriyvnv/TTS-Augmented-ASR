# syntts_asr — Qwen3-ASR fine-tuning container
#
# Base image choice:
#   nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
#   - nvcc 12.8 matches torch 2.10.0+cu128 exactly (no mismatch → flash-attn builds)
#   - -devel variant has compilers + headers (required to build flash-attn)
#   - cuDNN is bundled (safer for attention ops; torch ships its own but system libs
#     can take precedence in some code paths)
#   - Ubuntu 22.04 (stable, wide tool availability)
#
# Why this fixes what the host couldn't:
#   - Host has nvcc 13.0 (mismatch with torch cu12.8) → flash-attn build fails
#   - Host has no ffmpeg and no sudo → torchcodec needs PyAV symlink hack
#   - Inside this image: nvcc 12.8 + apt-installed ffmpeg → both solved
#
# Minimum driver: 565.x (released with CUDA 12.8). Host driver is 580.65.06 — OK.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# - ffmpeg: solves torchcodec audio decode (datasets>=3.0). No more av.libs symlinks.
# - build-essential, ninja: for compiling flash-attn kernels
# - tmux, less, vim-tiny: operator comfort inside a long-running container
# - git, curl, ca-certificates: cloning, uv installer
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      git \
      curl \
      ca-certificates \
      build-essential \
      ninja-build \
      tmux \
      screen \
      less \
      vim-tiny \
      htop \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv (install globally to /usr/local/bin so any user can invoke it)
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | \
      UV_INSTALL_DIR=/usr/local/bin UV_UNMANAGED_INSTALL=1 sh

# Python 3.12 via uv (matches host convention from CLAUDE.md)
RUN uv python install 3.12

# ---------------------------------------------------------------------------
# Project dependencies
# ---------------------------------------------------------------------------
# Put the venv OUTSIDE /workspace so runtime bind-mounts of the source tree
# don't mask it. UV_PROJECT_ENVIRONMENT + VIRTUAL_ENV point everything at /opt/venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# Copy only dependency manifests first so Docker caches the big deps layer
# unless pyproject.toml or uv.lock actually change.
COPY pyproject.toml uv.lock ./

# --no-install-project: install only the declared dependencies, skip installing
# the workspace project itself. The project source is bind-mounted from the
# host at runtime (into /workspace), so the image doesn't need src/ or README.md
# baked in — and `python -m src.training.train_qwen3_asr` works from CWD.
RUN uv sync --python 3.12 --locked --no-install-project

# ---------------------------------------------------------------------------
# flash-attention 2 (compiles inside image where nvcc == torch CUDA version)
# ---------------------------------------------------------------------------
# MAX_JOBS=4 caps concurrent C++ jobs (full parallelism OOMs on boxes with
# <64 GB RAM during compile). Takes ~15–20 min; only runs when this layer
# is rebuilt (cached otherwise).
#
# Using `uv pip install` (not `uv add`) so flash-attn does NOT modify uv.lock
# on the host when the workspace is bind-mounted at runtime.
#
# Hard-fail if the build doesn't work — the user expects flash-attn to be
# available, and a silent sdpa fallback would make that invisible.
RUN MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

# ---------------------------------------------------------------------------
# End-of-build sanity check: confirm all components that matter at training
# time actually import together. Fails the build if any are broken.
# ---------------------------------------------------------------------------
RUN python -c "\
import torch, flash_attn, transformers, datasets, qwen_asr; \
print(f'torch {torch.__version__} (CUDA {torch.version.cuda})'); \
print(f'flash_attn {flash_attn.__version__}'); \
print(f'transformers {transformers.__version__}'); \
print(f'datasets {datasets.__version__}'); \
print(f'qwen_asr OK'); \
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration; \
print('Qwen3ASRForConditionalGeneration importable'); \
"

# ---------------------------------------------------------------------------
# FFmpeg shared libs: apt-installed ffmpeg puts libs in /usr/lib/x86_64-linux-gnu
# which is already on the linker path. No LD_LIBRARY_PATH hack needed.
# ---------------------------------------------------------------------------

# Default: stay alive so the user can `docker exec` in whenever they want.
# Training is launched explicitly via scripts/docker/train.sh.
CMD ["sleep", "infinity"]
