#!/bin/bash
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           Verify Sweet Spot Evaluation Results                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if evaluation log exists
echo "1. Checking if evaluation log exists..."
if [ -f "logs/eval_sweet_spot_greedy.log" ]; then
    echo "   ✓ Found: logs/eval_sweet_spot_greedy.log"
    LOG_SIZE=$(wc -l < logs/eval_sweet_spot_greedy.log)
    echo "   Log file size: $LOG_SIZE lines"
else
    echo "   ✗ NOT FOUND: logs/eval_sweet_spot_greedy.log"
    echo "   Did the evaluation actually run?"
    exit 1
fi

echo ""
echo "2. Looking for pass@1 result in log..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Search for pass@1 patterns
grep -i "pass@1\|estimated pass\|final.*accuracy" logs/eval_sweet_spot_greedy.log | head -10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "3. Checking for errors in evaluation..."
if grep -qi "error\|exception\|failed\|traceback" logs/eval_sweet_spot_greedy.log; then
    echo "   ⚠️  Found errors in log:"
    grep -i "error\|exception\|failed" logs/eval_sweet_spot_greedy.log | head -5
else
    echo "   ✓ No obvious errors found"
fi

echo ""
echo "4. Checking checkpoint loading..."
grep -i "checkpoint\|loading.*model\|adapter" logs/eval_sweet_spot_greedy.log | head -10

echo ""
echo "5. Last 30 lines of evaluation log:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -30 logs/eval_sweet_spot_greedy.log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "POSSIBLE ISSUES:"
echo "  1. Checkpoint didn't load (would show as base model = 67.7%)"
echo "  2. Evaluation script crashed partway through"
echo "  3. Parser couldn't find the pass@1 line (returned None → default 67.7%)"
echo "  4. Model actually did get exactly 67.7% (very unlikely!)"
echo ""
echo "ACTION: Check the output above to diagnose which issue occurred."
