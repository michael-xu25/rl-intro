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
sudo apt-get install -y -qq build-essential ninja-build git wget

# ── 2. Upgrade pip ─────────────────────────────────────────────────────────
echo ""
echo ">>> [2/6] Upgrading pip ..."
pip install --upgrade pip setuptools wheel

# ── 3. Fix pre-installed packages that break with numpy 2.x ───────────────
#    Lightning AI ships old scipy/scikit-learn/pandas compiled against numpy 1.x.
#    These crash at import time with numpy 2.x ("dtype size changed").
echo ""
echo ">>> [3/6] Upgrading numpy/scipy/scikit-learn/pandas for compat ..."
pip install --upgrade numpy scipy scikit-learn pandas

# ── 4. Install flash-attn (build from source) ─────────────────────────────
#    openrlhf imports flash_attn at module level (can't skip it).
#    Pre-built wheels have ABI mismatch with Lightning's torch 2.8.0+cu128.
#    Must compile from source against the exact torch installation.
#
#    Requires: nvcc (CUDA compiler) + CUDA dev headers.
#    Lightning AI has the runtime but not dev tools, so install via conda.
echo ""
echo ">>> [4/6] Installing flash-attn (compiling from source, ~5-10 min) ..."

# Detect torch's CUDA version so we install the matching toolkit.
# torch 2.8.0+cu128 -> CUDA 12.8 -> conda pin ">=12.8,<12.9"
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
CUDA_PIN=">=${TORCH_CUDA},<$(echo "$TORCH_CUDA" | awk -F. '{print $1"."$2+1}')"
echo "    torch CUDA: $TORCH_CUDA -> conda pin: $CUDA_PIN"

# Install CUDA dev toolkit pinned to match torch exactly
echo "    Installing CUDA ${TORCH_CUDA} toolkit (nvcc + dev headers) ..."
conda install -y -c nvidia \
    "cuda-nvcc${CUDA_PIN}" \
    "cuda-cudart-dev${CUDA_PIN}" \
    "cuda-libraries-dev${CUDA_PIN}" \
    2>&1

# Set build environment
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

echo "    CUDA_HOME=$CUDA_HOME"
echo "    nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo 'NOT FOUND')"
echo "    torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "    Compiling flash-attn from source (~5-10 min) ..."

# FLASH_ATTENTION_FORCE_BUILD=TRUE forces source compilation
#   (skips pre-built wheel download which has ABI mismatch)
# --no-build-isolation lets it find the system torch
# --no-cache-dir avoids stale cache
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
    pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

unset TMPDIR

echo "    flash-attn installed successfully."

# ── 5. Install OpenRLHF + all dependencies ────────────────────────────────
echo ""
echo ">>> [5/6] Installing OpenRLHF + dependencies ..."

# openrlhf --no-deps to avoid pip trying to build flash-attn again
# (it's already installed from the wheel above)
pip install openrlhf --no-deps

# All remaining openrlhf deps with correct version pins
pip install \
    accelerate \
    bitsandbytes \
    deepspeed \
    einops \
    isort \
    jsonlines \
    loralib \
    optimum \
    "optree>=0.15.0" \
    peft \
    pylatexenc \
    "pynvml>=12.0.0" \
    "ray[default]==2.48.0" \
    sentencepiece \
    tokenizers \
    torchdata \
    transformers \
    transformers_stream_generator \
    vllm \
    wandb \
    datasets \
    lightning-sdk

# Verify critical imports
echo "    Verifying imports ..."
python3 -c "
import flash_attn
import openrlhf
import ray
import vllm
import transformers
import deepspeed
import sklearn
import scipy
print('    All imports OK.')
"

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
echo ""
echo "  Next step:"
echo "    WANDB_TOKEN=your_token bash src/train_math.sh"
echo "============================================"
echo ""
echo "Full log saved to: $LOGFILE"
