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
echo ">>> [1/5] Installing system build dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build git

# ── 2. Upgrade pip ─────────────────────────────────────────────────────────
echo ""
echo ">>> [2/5] Upgrading pip ..."
pip install --upgrade pip setuptools wheel

# ── 3. Fix pre-installed packages that break with numpy 2.x ───────────────
#    Lightning AI ships old scipy/scikit-learn/pandas compiled against numpy 1.x.
#    These crash at import time with numpy 2.x ("dtype size changed" / "multiarray
#    failed to import"). Upgrade them all in one shot.
echo ""
echo ">>> [3/5] Upgrading numpy/scipy/scikit-learn/pandas for compat ..."
pip install --upgrade numpy scipy scikit-learn pandas

# ── 4. Install OpenRLHF + all dependencies ────────────────────────────────
#    openrlhf has flash-attn==2.8.3 as a hard dep, which fails to build on
#    Lightning AI (cross-device link error). We install with --no-deps and
#    manually install every other dependency.
#
#    We use PyTorch's built-in SDPA attention (--attn_implementation sdpa)
#    instead of flash-attn. SDPA uses the same flash-attention CUDA kernels
#    under the hood on L4/A100 GPUs.
echo ""
echo ">>> [4/5] Installing OpenRLHF + dependencies ..."

# Step A: openrlhf itself (skip its deps to avoid flash-attn build)
pip install openrlhf --no-deps

# Step B: all openrlhf deps EXCEPT flash-attn, with correct version pins
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

# Step C: verify the critical imports work
echo "    Verifying imports ..."
python3 -c "
import openrlhf
import ray
import vllm
import transformers
import deepspeed
import sklearn
import scipy
print('    All imports OK.')
"

# ── 5. Pre-download model & dataset ───────────────────────────────────────
echo ""
echo ">>> [5/5] Pre-downloading model and dataset ..."
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
echo "  Attention: PyTorch SDPA (no flash-attn needed)"
echo ""
echo "  Next step:"
echo "    WANDB_TOKEN=your_token bash src/train_math.sh"
echo "============================================"
echo ""
echo "Full log saved to: $LOGFILE"
