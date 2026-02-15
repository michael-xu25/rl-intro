#!/usr/bin/env bash
# ============================================================================
# Tiny-Math-Solver — Entity-Tracking GRPO on GSM8K with TRL
# ============================================================================
# Usage:
#     bash src/train_math.sh
#     WANDB_TOKEN=your_token bash src/train_math.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# Step 1: Build curated entity-tracking dataset (if not already built)
if [ ! -d "data/entity_tracking_dataset" ]; then
    echo ">>> Building entity-tracking dataset from GSM8K ..."
    python src/build_entity_dataset.py
    echo ""
fi

# Step 2: Launch training
echo ">>> Launching GRPO training ..."
echo "    Model:   Qwen/Qwen2.5-1.5B-Instruct"
echo "    Dataset: GSM8K (filtered to 3+ entity problems)"
echo "    Method:  GRPO + LoRA (rank 16) + <think> tags"
echo "    Rewards: correctness (0/1) + intermediate step accuracy (0-0.5)"
echo ""

python src/train_grpo.py

echo ""
echo ">>> Done! Model saved to ./checkpoint/"
