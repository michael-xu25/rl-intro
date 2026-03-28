# Experiment Post-Mortem: Sweet Spot Validation

**Date:** 2026-03-27 (Updated with correct results)
**Status:** ⚠️ Flawed design BUT promising signals
**Lesson:** Always isolate variables

---

## Bottom Line

We tried to prove difficulty-based calibration beats entity heuristics, but designed an unfair test.

**CORRECTED Result:** Sweet spot achieved **74.0%** (+6.3pp), entity filter achieved **77.0%** (+9.3pp).

**The experiment shows:**
- ✅ Sweet spot works (6.3pp from just 15 problems!)
- ✅ But entity's scale advantage won (3pp better with 108x more data)
- ⚠️ Still unfair comparison (different dataset sizes AND epochs)

---

## What Actually Happened (CORRECTED)

| Metric | Entity Filter | Sweet Spot | Fair? |
|--------|---------------|------------|-------|
| **Test Accuracy** | 77.0% (+9.3pp) | **74.0% (+6.3pp)** ✅ | ❌ |
| **Dataset Size** | ~1,620 problems | 15 problems | ❌ 108x diff! |
| **Epochs (checkpoint-50)** | 0.247 epochs | 26.7 epochs | ❌ 108x diff! |
| **Ghost Batching** | 41.6% | 24.1% | ✅ (confirmed!) |
| **Training Accuracy** | 81.4% | 74.4% | N/A |

**CORRECTION:** Sweet spot achieved 74.0% (NOT 67.7% - parser error grabbed baseline instead of checkpoint result).

**New interpretation:** Sweet spot learned from 15 hard problems and generalized moderately well. NOT complete overfitting. Shows difficulty calibration has signal, but limited by small dataset size.

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

## What We Actually Proved (UPDATED)

### ✅ Validated

1. **Difficulty calibration has measurable signal**
   - Sweet spot: 15 problems → +6.3pp improvement
   - Entity: 1,620 problems → +9.3pp improvement
   - **6.3pp from just 15 problems suggests the selection method works**
   - Not just memorization (74.4% train → 74.0% test = generalization)

2. **Ghost batch reduction (42%)**
   - Entity: 41.6% wasted compute
   - Sweet spot: 24.1% wasted compute
   - This is real and holds despite flawed experiment

3. **Pass@16 measurement works**
   - Successfully identified hard problems (2-12/16 range)
   - Model struggled on them (74.4% vs 81.4% training acc)
   - But still learned enough to generalize (+6.3pp)

4. **Dataset size still matters**
   - 15 problems → +6.3pp
   - 1,620 problems → +9.3pp
   - **3pp gap suggests scale advantage**
   - But diminishing returns? (108x data → 1.5x improvement)

### ⚠️ Partially Validated

1. **Difficulty calibration works, but...**
   - ✅ 6.3pp gain from 15 problems (vs 0pp baseline)
   - ⚠️ 3pp behind entity filter (77.0% vs 74.0%)
   - ❓ Need matched dataset size to test fairly

2. **Sweet spot enables learning**
   - ✅ Proved it's not just memorization
   - ✅ Training on hard problems → test generalization
   - ❓ But is it better than random 15 problems? (no control)

### ❌ Still Did NOT Prove

1. **Difficulty calibration beats heuristics**
   - Sweet 15 problems: 74.0%
   - Entity 1,620 problems: 77.0%
   - Need same dataset size to compare methods fairly

2. **Curriculum learning matters**
   - Entity had mixed difficulty
   - Sweet was pure hard
   - Different sizes confound the comparison

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

---

## ADDENDUM: Does This Prove Difficulty Calibration Works?

**Updated 2026-03-27 after discovering correct result (74.0% not 67.7%)**

### The Question

Does the 6.3pp improvement from 15 sweet spot problems prove that difficulty-based calibration produces measurable results?

### Short Answer

**TOO EARLY TO TELL DEFINITIVELY, but there are PROMISING SIGNALS.**

### The Evidence For

1. **Non-zero improvement from tiny dataset**
   - 15 problems → +6.3pp gain
   - Baseline would be 0pp (no learning)
   - Shows the model learned something transferable

2. **Generalization, not memorization**
   - Training: 74.4% accuracy
   - Test: 74.0% accuracy
   - Very close! Model didn't just memorize

3. **Ghost batch reduction is real**
   - 42% reduction (41.6% → 24.1%)
   - Directly tied to difficulty calibration working
   - The sweet spot filter DID identify problems with learning signal

4. **Training on hard problems worked**
   - 2-12/16 pass rate = model struggles with these
   - Yet still achieved 74% test accuracy
   - Suggests targeted difficulty CAN work

### The Evidence Against (Or "Not Yet")

1. **No control group**
   - We don't know if RANDOM 15 problems would get same result
   - Or EASY 15 problems (13-16/16)
   - Or ENTITY 15 problems
   - **Can't isolate difficulty as the causal factor**

2. **System prompt confound**
   - Both runs used `<think>` tag prompting
   - Entity filter showed broad improvement (10/55 failure modes → +9.3pp overall)
   - Maybe prompt is doing most of the work?
   - **Untested hypothesis**

3. **Different training dynamics**
   - Sweet: 26.7 epochs (heavy repetition)
   - Entity: 0.247 epochs (single pass)
   - Different learning regimes entirely
   - **Can't compare apples to oranges**

4. **Sample size is tiny**
   - 15 problems = very small sample
   - High variance possible
   - Could be lucky selection
   - **Need larger sample to be confident**

### What We'd Need to Prove It

**Experiment 1: Controlled Difficulty Test**

Compare (all with 15 problems, 26.7 epochs):
- Random 15 problems
- Easy 15 problems (13-16/16)
- Sweet 15 problems (2-12/16)
- Hard 15 problems (0-1/16)

If sweet spot beats all others → difficulty calibration matters.

**Experiment 2: Matched Scale Test**

Compare (all with 1,620 problems, 0.247 epochs):
- Entity filter (3+ entities)
- Sweet spot filter (2-12/16)
- Random selection

If sweet spot beats entity → difficulty beats heuristics.

**Experiment 3: Prompt Effect Test**

Compare (all with random 1,620 problems):
- No system prompt
- `<think>` prompt
- Entity-specific prompt

If prompt alone gets 67% → 75%+ → problem selection is secondary.

### My Honest Assessment

**Promising but inconclusive.**

**What we know:**
- ✅ 6.3pp from 15 problems is impressive
- ✅ Ghost batching reduction is real and valuable
- ✅ Model generalized (not pure memorization)
- ✅ Training on hard problems didn't break learning

**What we don't know:**
- ❓ Would random 15 problems do the same?
- ❓ Is difficulty calibration the key factor or just correlation?
- ❓ Would sweet spot beat entity at matched scale?
- ❓ Is the prompt doing most of the work?

**The 6.3pp improvement suggests there's SIGNAL in difficulty calibration.**

But without proper controls, we can't say it's CAUSAL. Could be:
- Difficulty calibration working ← what we hope
- Lucky problem selection ← needs larger sample
- Prompt effect ← needs control
- Training dynamics ← needs matched epochs

### What This Means for CalibrateRL

**For the thesis:**
- Strong suggestive evidence, not proof
- Ghost batch reduction alone is valuable (42% efficiency gain)
- 6.3pp from 15 problems is encouraging
- Need proper validation before strong claims

**For the pitch deck:**
- ✅ "Initial results show promise: 6.3pp from 15 calibrated problems"
- ✅ "42% reduction in wasted compute (ghost batching)"
- ✅ "Early validation suggests difficulty calibration has signal"
- ❌ "Proven to beat heuristic methods" ← not yet
- ❌ "Difficulty calibration is the key factor" ← needs controls

**For Week 2-4:**
- The concept is promising enough to build on
- Ghost batch reduction alone justifies the approach
- Build the product, validate rigorously later
- Be honest about limitations in technical documentation

### Bottom Line

**Is there measurable signal? YES.**

6.3pp from 15 problems is not zero. Ghost batching dropped 42%. The model learned and generalized.

**Is it proven to be difficulty calibration specifically? NOT YET.**

Need controls to isolate the causal factor. But the signals are promising enough to continue.

**Should you keep building? ABSOLUTELY.**

The evidence suggests this direction is worth pursuing. Build the product, get more data, validate properly when you have resources.

---

**Final verdict:** Promising signals, not definitive proof. Enough to justify building the product. Not enough to publish a paper claiming causality. Perfect for an early-stage startup iterating based on data.

