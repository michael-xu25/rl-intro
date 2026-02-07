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
echo ">>> [1/4] Installing system build dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build git

# ── 2. Upgrade pip ─────────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] Upgrading pip ..."
pip install --upgrade pip setuptools wheel

# ── 3. Install OpenRLHF + dependencies ─────────────────────────────────────
#    We use PyTorch's built-in SDPA attention (--attn_implementation sdpa)
#    instead of flash-attn. SDPA uses the same flash-attention CUDA kernels
#    under the hood, and avoids the painful flash-attn compilation on
#    Lightning AI. This means we can skip the flash-attn dependency entirely.
echo ""
echo ">>> [3/4] Installing Python requirements ..."
pip install wandb datasets lightning-sdk

# openrlhf has flash-attn==2.8.3 as a hard dep, which fails to build on
# Lightning AI. We skip it with --no-deps and install the real deps manually.
echo "    Installing openrlhf (skipping flash-attn) ..."
pip install openrlhf --no-deps
pip install accelerate bitsandbytes deepspeed einops peft transformers \
            tokenizers sentencepiece "ray[default]" vllm

# ── 4. Pre-download model & dataset ───────────────────────────────────────
echo ""
echo ">>> [4/4] Pre-downloading model and dataset ..."
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
echo "    bash src/train_math.sh"
echo "============================================"
echo ""
echo "Full log saved to: $LOGFILE"
