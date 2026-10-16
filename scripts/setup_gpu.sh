#!/bin/bash
# =============================================================================
# Environment setup for a Linux CUDA machine (tested on a single H100 80GB).
# =============================================================================

set -e

echo "[1/4] Checking environment..."
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[2/4] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
else
    echo "uv already installed: $(uv --version)"
fi

echo "[3/4] Installing Python packages..."
uv sync

echo "[4/4] Logins..."
uv run hf auth whoami > /dev/null 2>&1 || uv run hf auth login
uv run python -c "import wandb; assert wandb.api.api_key" > /dev/null 2>&1 || uv run wandb login

uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
import nemo.collections.asr as nemo_asr
print('NeMo ASR: OK')
import jiwer, datasets, transformers
print('jiwer / datasets / transformers: OK')
"
