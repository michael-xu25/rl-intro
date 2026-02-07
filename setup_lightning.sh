#!/usr/bin/env bash
# ============================================================================
# One-time setup script for Lightning AI Studio
# ============================================================================
# Run this ONCE after cloning the repo into your Lightning Studio:
#     bash setup_lightning.sh
#
# All output is also saved to setup.log for debugging.
# ============================================================================
set -euo pipefail

LOGFILE="$(pwd)/setup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================"
echo "  Tiny-Math-Solver — Lightning AI Setup"
echo "  $(date)"
echo "============================================"

# ── 1. System dependencies ─────────────────────────────────────────────────
echo ""
echo ">>> [1/6] Installing system build dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build git

# ── 2. Upgrade pip ─────────────────────────────────────────────────────────
echo ""
echo ">>> [2/6] Upgrading pip ..."
pip install --upgrade pip setuptools wheel

# ── 3. Install CUDA dev toolkit (nvcc) via conda ──────────────────────────
#    Lightning AI has the CUDA runtime but not nvcc / dev headers.
#    We always install via conda — it's fast and avoids path guessing.
echo ""
echo ">>> [3/6] Installing CUDA development toolkit via conda ..."
echo "    (This ensures nvcc is available for compiling flash-attn)"

# Detect CUDA version from torch to install matching toolkit
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.8")
echo "    Detected torch CUDA version: $TORCH_CUDA"

# Extract major.minor for conda package (e.g. 12.8 -> 12.8)
CUDA_MAJOR_MINOR="${TORCH_CUDA%.*}.${TORCH_CUDA#*.}"

conda install -y -c nvidia \
    cuda-nvcc \
    cuda-libraries-dev \
    cuda-cudart-dev \
    2>&1 || {
    echo "    WARNING: conda install of cuda toolkit failed, trying apt ..."
    sudo apt-get install -y nvidia-cuda-toolkit
}

# Set CUDA_HOME — prefer conda env, fallback to /usr/local/cuda
if [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
elif [ -f "/usr/local/cuda/bin/nvcc" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif command -v nvcc &>/dev/null; then
    export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
else
    echo "    ERROR: Could not find nvcc anywhere!"
    echo "    Will try to install openrlhf without flash-attn (using SDPA fallback)."
fi

export PATH="${CUDA_HOME:-}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-}/lib64:${LD_LIBRARY_PATH:-}"

echo "    CUDA_HOME = ${CUDA_HOME:-NOT SET}"
echo "    nvcc:      $(which nvcc 2>/dev/null && nvcc --version 2>/dev/null | tail -1 || echo 'NOT FOUND')"

# ── 4. Install Flash Attention 2 ──────────────────────────────────────────
#    flash-attn must be installed BEFORE openrlhf (which depends on it).
#    If compilation fails, we fall back to SDPA attention (built into PyTorch).
echo ""
echo ">>> [4/6] Installing flash-attn (may take 5-10 min to compile) ..."

FLASH_ATTN_OK=false

if command -v nvcc &>/dev/null; then
    MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation 2>&1 && FLASH_ATTN_OK=true || {
        echo ""
        echo "    flash-attn source build failed. Trying without version pin ..."
        MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>&1 && FLASH_ATTN_OK=true || true
    }
fi

if [ "$FLASH_ATTN_OK" = false ]; then
    echo ""
    echo "    ============================================================"
    echo "    WARNING: flash-attn installation failed."
    echo "    Falling back to PyTorch SDPA attention (still fast!)."
    echo "    Patching train_math.sh to use --attn_implementation sdpa"
    echo "    ============================================================"
    # Patch the training script to use sdpa instead of flash_attention_2
    sed -i 's/flash_attention_2/sdpa/g' src/train_math.sh
    # Remove flash_attn flags
    sed -i '/--flash_attn/d' src/train_math.sh
fi

# ── 5. Install OpenRLHF + remaining dependencies ──────────────────────────
echo ""
echo ">>> [5/6] Installing Python requirements ..."

if [ "$FLASH_ATTN_OK" = true ]; then
    pip install -r requirements.txt
else
    # Install openrlhf without letting pip try to build flash-attn again
    pip install wandb datasets lightning-sdk
    pip install openrlhf --no-build-isolation 2>&1 || {
        echo "    openrlhf install failed — trying without flash-attn dep ..."
        pip install openrlhf --no-deps
        # Manually install key openrlhf dependencies (minus flash-attn)
        pip install accelerate bitsandbytes deepspeed einops peft transformers \
                    tokenizers sentencepiece datasets ray[default] 2>&1 || true
    }
fi

# ── 6. Pre-download model & dataset ───────────────────────────────────────
echo ""
echo ">>> [6/6] Pre-downloading model and dataset ..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen/Qwen2.5-0.5B-Instruct tokenizer ...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('Downloading Qwen/Qwen2.5-0.5B-Instruct model ...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('Done.')
"

python3 -c "
from datasets import load_dataset
print('Downloading GSM8K ...')
load_dataset('openai/gsm8k', 'main', split='train')
print('GSM8K downloaded.')
"

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Setup complete!  $(date)"
if [ "$FLASH_ATTN_OK" = true ]; then
    echo "  flash-attn: INSTALLED (flash_attention_2)"
else
    echo "  flash-attn: SKIPPED (using SDPA fallback)"
fi
echo ""
echo "  Next step:"
echo "    bash src/train_math.sh"
echo "============================================"
echo ""
echo "Full log saved to: $LOGFILE"
