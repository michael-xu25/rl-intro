#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          Sweet Spot Checkpoint-50 Evaluation                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Find the checkpoint
CHECKPOINT=$(find checkpoint -name "checkpoint-50" -type d | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "❌ No checkpoint-50 found!"
    echo ""
    echo "Available checkpoints:"
    find checkpoint -name "checkpoint-*" -type d 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation 1: Greedy (fair comparison to baseline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EVALUATION 1: Greedy Decoding (temp=0, fair baseline comparison)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This matches the baseline evaluation methodology:"
echo "  - Temperature: 0 (deterministic)"
echo "  - Samples: 1 per problem"
echo "  - System prompt: YES (training prompt)"
echo "  - Expected time: ~10 minutes"
echo ""

python3 src/eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --greedy \
    2>&1 | tee logs/eval_sweet_spot_greedy.log

echo ""
echo "✓ Greedy evaluation complete"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation 2: Pass@16 (capability assessment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "EVALUATION 2: Pass@16 (capability assessment, temp=0.7)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This measures if RL improved latent capability:"
echo "  - Temperature: 0.7 (sampling)"
echo "  - Samples: 16 per problem"
echo "  - System prompt: YES (training prompt)"
echo "  - Expected time: ~30 minutes"
echo ""

python3 src/eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    2>&1 | tee logs/eval_sweet_spot_pass16.log

echo ""
echo "✓ Pass@16 evaluation complete"
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      EVALUATION COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - logs/eval_sweet_spot_greedy.log   (greedy pass@1)"
echo "  - logs/eval_sweet_spot_pass16.log   (pass@16 capability)"
echo ""
echo "Baseline comparison:"
echo "  Baseline:      67.7% greedy pass@1, 95% pass@16"
echo "  Entity filter: 77.0% greedy pass@1, 98% pass@16 (+9.3 pp)"
echo "  Sweet spot:    Check the logs above!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
