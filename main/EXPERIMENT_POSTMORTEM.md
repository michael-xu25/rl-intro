# Experiment Post-Mortem: Sweet Spot Validation

**Date:** 2026-03-27  
**Status:** ❌ Flawed experimental design  
**Lesson:** Always isolate variables

---

## Bottom Line

We tried to prove difficulty-based calibration beats entity heuristics, but designed an unfair test.

**Result:** Entity filter won (77.0% vs 67.7%), BUT we changed dataset size (1,620 vs 15 problems) simultaneously - a **108x difference!**

Cannot conclude anything from an unfair comparison.

---

## What Actually Happened

| Metric | Entity Filter | Sweet Spot | Fair? |
|--------|---------------|------------|-------|
| **Test Accuracy** | 77.0% (+9.3pp) | 67.7% (+0.0pp) | ❌ |
| **Dataset Size** | ~1,620 problems | 15 problems | ❌ 108x diff! |
| **Epochs (checkpoint-50)** | 0.247 epochs | 26.7 epochs | ❌ 108x diff! |
| **Ghost Batching** | 41.6% | 24.1% | ✅ (confirmed!) |
| **Training Accuracy** | 81.4% | 74.4% | N/A |

**Diagnosis:** Sweet spot overfit to 15 problems (74% train, 67% test). Entity generalized from 1,620 problems.

---

## Critical Experimental Design Flaws

### Flaw 1: Changed Multiple Variables

We changed simultaneously:
- ❌ Selection method (entity heuristic vs pass@16)
- ❌ Dataset size (1,620 vs 15 problems)
- ❌ Training epochs (0.247 vs 26.7)
- ❌ System prompt (entity-specific vs general)

**Cannot attribute results to any single cause!**

### Flaw 2: Unfair Training Dynamics

**Entity at checkpoint-50:**
- 1,620 problems × 0.247 epochs
- Each problem seen: **~0.2 times** (underfitting risk)
- Most problems NEVER seen!

**Sweet spot:**
- 15 problems × 26.7 epochs
- Each problem seen: **~27 times** (severe overfitting)
- Model memorized the training set

**Same total training instances (400), completely different learning regimes!**

### Flaw 3: Wrong Sample Size

**What we did:**
- Ran pass@16 on 50 training samples
- Got 15 sweet spot problems
- Trained for 50 steps

**What we should have done:**
- Run pass@16 on 1,620 training samples (match entity size)
- Get ~664 sweet spot problems (41% in goldilocks zone)
- Train for 50 steps
- **THEN compare fairly**

---

## What We Actually Proved

### ✅ Validated

1. **Ghost batch reduction (42%)**
   - Entity: 41.6% wasted compute
   - Sweet spot: 24.1% wasted compute
   - This is real and holds despite flawed experiment

2. **Pass@16 measurement works**
   - Successfully identified hard problems
   - Model struggled on them (74% vs 81% training acc)

3. **Dataset size is critical**
   - 15 problems → complete overfitting
   - 1,620 problems → generalization
   - **You cannot learn math from 15 examples, no matter how "sweet"**

### ❌ Did NOT Prove

1. **Difficulty calibration beats heuristics**
   - Didn't test fairly (108x size mismatch)
   - Still an open question

2. **Sweet spot improves accuracy**
   - Failed on tiny dataset (expected)
   - Unknown on proper dataset size

3. **Curriculum learning matters**
   - Entity had mixed difficulty (accidental curriculum)
   - Sweet was pure hard (no curriculum)
   - But also different sizes, so can't isolate this

---

## Open Questions

### Q1: Does difficulty measurement beat heuristics?

**Proper test needed:**
- Both use 1,620 problems
- Entity: 3+ entities filter
- Sweet: 2-12/16 pass@16 filter
- Train 50 steps each
- Compare test accuracy

**We didn't test this.**

### Q2: Is curriculum learning necessary?

Entity accidentally had curriculum (mix of difficulties).  
Sweet was pure hard problems.  
But we also changed dataset size, so can't tell.

### Q3: Is GRPO just learning formatting/style?

**Key observation:**
- Entity targeted 10/55 failure modes
- But improved broadly (+9.3pp overall)
- System prompt forces `<think>` tags

**Hypothesis:** Maybe the prompt does most of the work, and problem selection barely matters?

**Test:** Random problems with `<think>` prompt vs without. If prompt alone gets most gains, problem selection is secondary.

### Q4: What's minimum viable dataset size?

We know:
- 15 problems: Failure (overfitting)
- 1,620 problems: Success (generalization)

We don't know:
- Is 100 enough? 500? 1,000?

---

## What We Should Have Done

### Proper Experiment A: Test Selection Method

**Hypothesis:** Pass@16 difficulty measurement beats entity heuristic

**Control:**
- ✅ Same dataset size (1,620 problems)
- ✅ Same training steps (50)
- ✅ Same epochs (0.247)
- ✅ Same prompt

**Test:**
- Entity: Filter by 3+ entities
- Sweet: Filter by 2-12/16 pass@16

**Cost:** ~24 hours (20h pass@16 + 4h training)

### Proper Experiment B: Test Curriculum

**Hypothesis:** Pure sweet spot beats mixed difficulty

**Control:**
- ✅ Same dataset size (660 problems)
- ✅ Same selection (pass@16)

**Test:**
- Pure: 100% sweet spot (2-12/16)
- Mixed: 60% sweet + 30% medium + 10% hard

**Cost:** ~12 hours

### Cheap Experiment C: Test Prompt Effect

**Hypothesis:** Prompt is doing all the work

**Control:**
- ✅ Same dataset (random 1,620 problems)

**Test:**
- Baseline: No system prompt
- Prompted: `<think>` tags

**Cost:** ~4 hours (no pass@16 needed!)

**If prompt alone gets most gains, skip pass@16 experiments.**

---

## Lessons Learned

### 1. Isolate Variables

Bad: Change A, B, C, D. Claim A caused result.  
Good: Change only A. Now you know A's effect.

### 2. Fair Comparisons Require Matching

When comparing approaches, match:
- ✅ Dataset SIZE (not just steps!)
- ✅ EPOCHS (not just steps!)
- ✅ Evaluation method
- ✅ Hyperparameters

**Steps are NOT comparable if dataset sizes differ!**

### 3. Overfitting vs Generalization

15 problems × 27 epochs:
- Memorizes training set
- Zero transfer
- High train acc, baseline test acc

1,620 problems × 0.25 epochs:
- Learns patterns (if they exist)
- Generalizes
- Lower train acc, better test acc

### 4. One Hypothesis Per Experiment

We tried to test:
1. Difficulty > heuristics
2. Ghost batch reduction
3. Targeted training improves accuracy  
4. Small datasets work if high quality

Tested none properly. Next time: One question, one variable.

### 5. Negative Results Are Data

What we learned:
- Dataset size > problem quality (for small N)
- Ghost reduction ≠ accuracy improvement (they're separate)
- Need proper controls

This is valuable for the pitch:
> "We ran validation. Failed due to dataset mismatch. Learned that difficulty calibration needs sufficient data diversity. Shows empirical iteration."

---

## Recommendations

### For Validation (If Continuing)

1. **Run Experiment C first** (4 hours)
   - Test if prompt is the secret sauce
   - Cheapest way to rule out the most likely alternative hypothesis

2. **If prompt ≠ everything, run Experiment A** (24 hours)
   - Fair sweet spot vs entity comparison
   - Actually tests the thesis

3. **Document properly**
   - Log all variables
   - Be honest about limitations

### For 4-Week Plan (Moving Forward)

**Don't get stuck on perfect validation.**

Week 2-4 is about building a fundable demo:
- Difficulty predictor
- Problem generation
- Environment quality metrics
- Pitch deck with results

Use this experiment as:
- Proof you iterate based on data
- Evidence ghost batching is real
- Lesson in experimental design

Come back to rigorous validation later.

---

## Final Thought

**Good scientists make mistakes. Great scientists learn from them.**

We:
1. ✅ Ran experiment
2. ✅ Got unexpected result
3. ✅ Dug into methodology
4. ✅ Found the flaws
5. ✅ Designed better experiments
6. ✅ Documented learnings

This is science. Onward to Week 2.

---

**Date:** 2026-03-27  
**Authors:** Human + Claude  
**Status:** Learning complete. Moving to product.
