#!/usr/bin/env bash
# ============================================================================
# Tiny-Math-Solver — GRPO training on GSM8K with TRL
# ============================================================================
# Usage:
#     bash src/train_math.sh
#     WANDB_TOKEN=your_token bash src/train_math.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">>> Launching GRPO training ..."
echo "    Model:   Qwen/Qwen2.5-3B-Instruct"
echo "    Dataset: GSM8K"
echo "    Method:  GRPO + LoRA (rank 16)"
echo ""

python src/train_grpo.py

echo ""
echo ">>> Done! Model saved to ./checkpoint/"
