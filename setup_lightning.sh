#!/usr/bin/env bash
# ============================================================================
# One-time setup script for Lightning AI Studio
# ============================================================================
# Run this ONCE after cloning the repo into your Lightning Studio:
#     bash setup_lightning.sh
# ============================================================================
set -euo pipefail

echo "============================================"
echo "  Tiny-Math-Solver — Lightning AI Setup"
echo "============================================"

# ── 1. System dependencies (needed for flash-attn build) ───────────────────
echo ""
echo ">>> Installing system build dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential ninja-build git

# ── 2. Upgrade pip ──────────────────────────────────────────────────────────
echo ""
echo ">>> Upgrading pip ..."
pip install --upgrade pip setuptools wheel

# ── 3. Install Flash Attention 2 FIRST (needs torch in build env) ──────────
#    Must come before openrlhf because openrlhf[vllm] depends on
#    flash-attn==2.8.3 and pip cannot build it in an isolated env.
echo ""
echo ">>> Installing flash-attn (this may take 5-10 minutes) ..."
pip install flash-attn==2.8.3 --no-build-isolation

# ── 4. Install OpenRLHF + remaining dependencies ──────────────────────────
echo ""
echo ">>> Installing Python requirements ..."
pip install -r requirements.txt

# ── 5. Pre-download model & dataset so training starts immediately ──────────
echo ""
echo ">>> Pre-downloading Qwen/Qwen2.5-0.5B-Instruct ..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer ...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('Downloading model ...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('Done.')
"

echo ""
echo ">>> Pre-downloading GSM8K dataset ..."
python3 -c "
from datasets import load_dataset
load_dataset('openai/gsm8k', 'main', split='train')
print('GSM8K downloaded.')
"

echo ""
echo "============================================"
echo "  Setup complete!  Next step:"
echo "    bash src/train_math.sh"
echo "============================================"
