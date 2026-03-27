"""
Extract and display final comparison results.

Shows entity filter vs sweet spot side-by-side.
"""

import re
from pathlib import Path

def extract_pass_at_1(log_path):
    """Extract greedy pass@1 from eval log."""
    if not Path(log_path).exists():
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Look for the pass@1 result
    patterns = [
        r"Estimated pass@1:\s*([\d.]+)",
        r"Pass@1:\s*([\d.]+)",
        r"pass@1.*?(\d+\.\d+)%",
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            # Convert to fraction if it's a percentage
            if val > 1.0:
                val = val / 100.0
            return val

    return None


def main():
    print("=" * 80)
    print("FINAL RESULTS: Entity Filter vs Sweet Spot")
    print("=" * 80)
    print()

    # Known baselines
    baseline_pass1 = 0.677
    entity_pass1 = 0.770
    entity_delta = entity_pass1 - baseline_pass1

    # Try to load sweet spot results
    sweet_greedy_log = "logs/eval_sweet_spot_greedy.log"
    sweet_pass1 = extract_pass_at_1(sweet_greedy_log)

    print("GREEDY PASS@1 (Test Set Performance):")
    print("-" * 80)
    print(f"{'Approach':<25} {'Pass@1':<15} {'Delta vs Baseline':<20}")
    print("-" * 80)
    print(f"{'Baseline (no training)':<25} {baseline_pass1*100:>6.1f}%        {'—':<20}")
    print(f"{'Entity filter':<25} {entity_pass1*100:>6.1f}%        {entity_delta*100:>+6.1f} pp")

    if sweet_pass1 is not None:
        sweet_delta = sweet_pass1 - baseline_pass1
        print(f"{'Sweet spot':<25} {sweet_pass1*100:>6.1f}%        {sweet_delta*100:>+6.1f} pp")

        print()
        print("=" * 80)
        print("DIRECT COMPARISON: Entity vs Sweet Spot")
        print("=" * 80)
        print()

        comparison = sweet_pass1 - entity_pass1

        print(f"Entity filter:  {entity_pass1*100:.1f}%")
        print(f"Sweet spot:     {sweet_pass1*100:.1f}%")
        print(f"Difference:     {comparison*100:+.1f} percentage points")
        print()

        if comparison > 0:
            print("✅ SWEET SPOT WINS!")
            print(f"   Better generalization despite lower training accuracy")
            print(f"   Training: 74.4% vs 81.4% (entity)")
            print(f"   Test:     {sweet_pass1*100:.1f}% vs {entity_pass1*100:.1f}% (entity)")
        elif comparison < -0.02:
            print("⚠️  Entity filter wins on final accuracy")
            print(f"   But sweet spot still won on efficiency:")
            print(f"   Ghost batching: 24.1% vs 41.6% (42% reduction)")
        else:
            print("✅ COMPARABLE PERFORMANCE")
            print(f"   Sweet spot matches entity with better efficiency:")
            print(f"   Ghost batching: 24.1% vs 41.6% (42% reduction)")

        print()
        print("=" * 80)
        print("TRAINING EFFICIENCY SUMMARY")
        print("=" * 80)
        print()

        print("Entity Filter:")
        print(f"  Training accuracy:  81.4%")
        print(f"  Test accuracy:      {entity_pass1*100:.1f}%")
        print(f"  Ghost batching:     41.6%")
        print(f"  Effective compute:  58.4%")
        print()

        print("Sweet Spot:")
        print(f"  Training accuracy:  74.4%")
        print(f"  Test accuracy:      {sweet_pass1*100:.1f}%")
        print(f"  Ghost batching:     24.1%")
        print(f"  Effective compute:  75.9%")
        print()

        efficiency_gain = (75.9 - 58.4) / 58.4 * 100
        print(f"Compute efficiency improvement: {efficiency_gain:+.0f}%")

        print()
        print("=" * 80)
        print("VERDICT")
        print("=" * 80)
        print()

        if sweet_pass1 >= entity_pass1 - 0.02:  # Within 2pp
            print("✅ THESIS VALIDATED")
            print()
            print("Difficulty-based calibration (sweet spot) achieves comparable or")
            print("better results with significantly less wasted compute.")
            print()
            print("Key insights:")
            print("  1. 42% reduction in ghost batching (41.6% → 24.1%)")
            print("  2. 30% more effective compute utilization")
            print("  3. Training on harder problems → better/equal generalization")
            print()
            print("Next steps:")
            print("  - Scale up: 500 training samples, 500 training steps")
            print("  - Build difficulty predictor (Week 2)")
            print("  - Problem generation pipeline (Week 2)")
        else:
            print("⚠️  MIXED RESULTS")
            print()
            print("Sweet spot improved efficiency but didn't match final accuracy.")
            print("This suggests the current sweet spot dataset is too small/hard.")
            print()
            print("Possible improvements:")
            print("  - Larger sweet spot sample (100-200 problems instead of 50)")
            print("  - Widen sweet spot range (1-13/16 instead of 2-12/16)")
            print("  - Train longer (100-200 steps instead of 50)")
            print("  - Combine approaches: sweet spot + some easier problems")

    else:
        print(f"{'Sweet spot':<25} {'PENDING':<15} {'Run evaluation!':<20}")
        print()
        print("Run: chmod +x run_evaluation.sh && ./run_evaluation.sh")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
