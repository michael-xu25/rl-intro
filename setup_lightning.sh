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

# ── 3. Install Flash Attention 2 (pre-built wheel) ────────────────────────
#    flash-attn must be installed BEFORE openrlhf (which depends on it).
#
#    Lightning AI has a cross-device filesystem issue that breaks
#    flash-attn's built-in wheel downloader (Errno 18: Invalid cross-device
#    link). The fix: download the pre-built wheel directly from GitHub and
#    install it with pip — no compilation needed.
echo ""
echo ">>> [3/5] Installing flash-attn (pre-built wheel) ..."

FLASH_ATTN_OK=false

# Detect torch/CUDA/Python to pick the right wheel
TORCH_VER=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:2]))")
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "    torch=$TORCH_VER  python=$PY_VER  platform=linux_x86_64"

# Pre-built wheel URL from flash-attn GitHub releases
# Format: flash_attn-{ver}+cu12torch{torch_ver}cxx11abiTRUE-{py}-{py}-linux_x86_64.whl
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TORCH_VER}cxx11abiTRUE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
echo "    Wheel URL: $WHEEL_URL"

# Strategy 1: Install pre-built wheel directly (fastest, no compilation)
pip install "$WHEEL_URL" 2>&1 && FLASH_ATTN_OK=true || {
    echo "    Pre-built wheel failed. Trying with TMPDIR fix ..."
    # Strategy 2: Fix the cross-device link error by putting TMPDIR on same FS
    export TMPDIR="$HOME/tmp"
    mkdir -p "$TMPDIR"
    MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir 2>&1 && FLASH_ATTN_OK=true || {
        echo "    Source build also failed."
    }
    unset TMPDIR
}

if [ "$FLASH_ATTN_OK" = false ]; then
    echo ""
    echo "    ============================================================"
    echo "    WARNING: flash-attn installation failed."
    echo "    Falling back to PyTorch SDPA attention (still fast!)."
    echo "    Patching train_math.sh to use --attn_implementation sdpa"
    echo "    ============================================================"
    sed -i 's/flash_attention_2/sdpa/g' src/train_math.sh
    sed -i '/--flash_attn/d' src/train_math.sh
fi

# ── 4. Install OpenRLHF + remaining dependencies ──────────────────────────
echo ""
echo ">>> [4/5] Installing Python requirements ..."

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
