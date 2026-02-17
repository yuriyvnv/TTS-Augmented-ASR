#!/bin/bash
# =============================================================================
# Setup script for Parakeet/Whisper fine-tuning on Vast.ai H100
# =============================================================================

set -e

echo "======================================"
echo "ASR Fine-Tuning Setup"
echo "======================================"

# 1. Check Python & CUDA
echo -e "\n[1/6] Checking environment..."
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Install UV
echo -e "\n[2/6] Installing UV package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "UV already installed: $(uv --version)"
fi

# 3. Install dependencies
echo -e "\n[3/6] Installing Python packages..."
uv sync
uv pip install 'nemo_toolkit[asr]'

# 4. Install screen
echo -e "\n[4/6] Installing screen..."
if ! command -v screen &> /dev/null; then
    apt update && apt install -y screen
else
    echo "Screen already installed"
fi

# 5. HuggingFace login
echo -e "\n[5/6] Setting up HuggingFace..."
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "Already logged in as: $(huggingface-cli whoami 2>/dev/null | head -1)"
else
    huggingface-cli login
fi

# 6. WandB login
echo -e "\n[6/6] Setting up WandB..."
if python3 -c "import wandb; assert wandb.api.api_key" > /dev/null 2>&1; then
    echo "WandB already configured"
else
    wandb login
fi

# Verify everything
echo ""
echo "======================================"
echo "Verification"
echo "======================================"
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
import nemo.collections.asr as nemo_asr
print('NeMo ASR: OK')
import wandb
print('WandB: OK')
from huggingface_hub import HfApi
print('HF Hub: OK')
from datasets import load_dataset
print('Datasets: OK')
import jiwer
print('jiwer: OK')
print()
print('All good. Run: ./scripts/train_parakeet.sh')
"
