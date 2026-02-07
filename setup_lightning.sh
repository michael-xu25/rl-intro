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

# ── 3. Set up CUDA toolkit (nvcc + headers needed to compile flash-attn) ───
#    Lightning AI ships the CUDA runtime but not the dev toolkit.
echo ""
echo ">>> Setting up CUDA development toolkit ..."

# Try to find an existing CUDA installation
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif [ -d "/usr/lib/cuda" ]; then
    export CUDA_HOME="/usr/lib/cuda"
else
    # Install CUDA dev toolkit via conda (Lightning uses conda envs)
    echo "    No CUDA_HOME found — installing cuda-nvcc via conda ..."
    conda install -y -c nvidia cuda-nvcc cuda-libraries-dev cuda-cudart-dev
    export CUDA_HOME="$CONDA_PREFIX"
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "    CUDA_HOME = $CUDA_HOME"
echo "    nvcc:      $(which nvcc 2>/dev/null || echo 'not found — will install')"

# If nvcc still isn't found after the above, install via conda as fallback
if ! command -v nvcc &>/dev/null; then
    echo "    nvcc still missing — installing via conda ..."
    conda install -y -c nvidia cuda-nvcc cuda-libraries-dev cuda-cudart-dev
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "    nvcc now at: $(which nvcc)"
fi

# ── 4. Install Flash Attention 2 FIRST (needs torch in build env) ──────────
#    Must come before openrlhf because openrlhf[vllm] depends on
#    flash-attn==2.8.3 and pip cannot build it in an isolated env.
echo ""
echo ">>> Installing flash-attn (this may take 5-10 minutes) ..."
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation

# ── 5. Install OpenRLHF + remaining dependencies ──────────────────────────
echo ""
echo ">>> Installing Python requirements ..."
pip install -r requirements.txt

# ── 6. Pre-download model & dataset so training starts immediately ──────────
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
