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

# ── 4. Install flash-attn from pre-built wheel ────────────────────────────
#    openrlhf imports flash_attn at module level (can't skip it).
#    pip install from URL fails on Lightning due to cross-device link error
#    (Errno 18: /tmp and /home are on different mounts).
#    Fix: wget the wheel to $HOME (same FS as pip), install from local file.
echo ""
echo ">>> [4/6] Installing flash-attn (pre-built wheel) ..."

TORCH_VER=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
WHEEL_NAME="flash_attn-2.8.3+cu12torch${TORCH_VER}cxx11abiTRUE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME}"

echo "    torch=$TORCH_VER  python=$PY_VER"
echo "    Downloading: $WHEEL_NAME"

cd "$HOME"
wget -q "$WHEEL_URL" -O "$WHEEL_NAME"
pip install "./$WHEEL_NAME"
rm -f "$WHEEL_NAME"
cd -

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
