"""
Rule-Based Reward Functions for GSM8K Math Problems.

Designed from Pass@16 analysis of Qwen2.5-1.5B-Instruct:
- pass@1: 67.7%, pass@16: 95%
- 33 problems in 1-10/16 range (RL sweet spot)
- Dominant error: comprehension (misreading which number modifies which noun)
- GSM8K gold labels contain <<expr=result>> intermediate step annotations

Five reward functions (thematic approach from gap analysis):
  1. correctness_reward:   1.0 if final answer matches gold, 0.0 otherwise
  2. reasoning_reward:     0.0-0.5 partial credit for correct intermediate steps
  3. format_reward:        0.0-0.2 for structured step-by-step reasoning
  4. hallucination_penalty: -0.3-0.0 penalty for introducing ungrounded numbers
  5. repetition_penalty:   -0.2-0.0 penalty for repeated reasoning patterns

Total reward range: -0.5 to 1.7, with correctness always dominant.
"""

import re
import logging
from collections import Counter

logger = logging.getLogger("tiny_math_solver")

_step_counter = 0
LOG_EVERY = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_text(completion) -> str:
    """Extract plain text from a completion (handles both str and chat format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        return ""
    return str(completion)


def _normalize_number(s: str) -> float | None:
    """Convert a string number to float for numeric comparison."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _numbers_match(a: str, b: str) -> bool:
    """Numeric comparison: '38.00' == '38', '1,000' == '1000'."""
    a_f = _normalize_number(a)
    b_f = _normalize_number(b)
    if a_f is None or b_f is None:
        return False
    return abs(a_f - b_f) < 1e-6


def _extract_all_numbers(text: str) -> set[float]:
    """Extract all numeric values from text as floats.

    Finds patterns like: 750, 1,200, 36.5, $4,400
    Returns a set of float values for numeric comparison.
    """
    raw = re.findall(r"[\d,]+(?:\.\d+)?", text)
    result = set()
    for r in raw:
        val = _normalize_number(r)
        if val is not None:
            result.add(val)
    return result


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_gold_answer(label: str) -> str | None:
    """Extract the number after '####' in a GSM8K answer."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    """Extract the model's final answer. Returns (answer, method)."""
    # 1. \boxed{<number>} -- model's preferred format
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", ""), "boxed"

    # 2. #### <number>
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", ""), "####"

    # 3. "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", ""), "the_answer_is"

    # 4. **<number>**
    matches = re.findall(r"\*\*([\d,]+(?:\.\d+)?)\*\*", text)
    if matches:
        return matches[-1].replace(",", ""), "bold"

    # 5. Last number in text (fallback)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number_fallback"

    return None, "no_answer"


# ── Gold step extraction (for reasoning reward) ─────────────────────────────

def extract_gold_steps(gold_label: str) -> list[float]:
    """Extract intermediate calculation results from GSM8K gold labels.

    GSM8K format: "320+430=<<320+430=750>>750 dollars."
    The <<expression=result>> markers contain intermediate computation results.

    Returns list of floats: [750.0, 700.0, 2280.0]
    """
    raw = re.findall(r"<<[^=]+=([^>]+)>>", gold_label)
    results = []
    for r in raw:
        val = _normalize_number(r.strip())
        if val is not None:
            results.append(val)
    return results


def extract_question_numbers(question: str) -> set[float]:
    """Extract all numbers that appear in the question text.

    Used to filter out intermediate step results that are just restating
    input values (no evidence the model actually computed them).
    """
    return _extract_all_numbers(question)


# ── Reward function 1: Correctness ──────────────────────────────────────────

def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """
    Core reward: 1.0 if final answer is numerically correct, 0.0 otherwise.
    """
    global _step_counter
    _step_counter += 1

    rewards = []
    samples = []
    for completion, gold_label in zip(completions, answer):
        text = _get_text(completion)
        gold = extract_gold_answer(gold_label)
        pred, method = extract_predicted_answer(text)
        correct = gold is not None and pred is not None and _numbers_match(pred, gold)
        rewards.append(1.0 if correct else 0.0)
        samples.append((text, gold, pred, method, correct))

    # Debug logging
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        n_correct = int(sum(rewards))
        n_parsed = sum(1 for _, _, pred, _, _ in samples if pred is not None)
        print(flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  CORRECTNESS (step {_step_counter}) — "
              f"{n_correct}/{len(rewards)} correct, "
              f"{n_parsed}/{len(rewards)} parsed", flush=True)
        print(f"{'='*70}", flush=True)
        for i, (text, gold, pred, method, correct) in enumerate(samples[:4]):
            snippet = text[:500].replace('\n', ' | ')
            if len(text) > 500:
                snippet += "..."
            status = "CORRECT" if correct else "WRONG  "
            print(f"  [{status}] gold={gold}  pred={pred}  method={method}", flush=True)
            print(f"  Output: {snippet}", flush=True)
            print(f"  ---", flush=True)
        print(f"{'='*70}\n", flush=True)

    return rewards


# ── Reward function 2: Reasoning (intermediate step verification) ────────────

def reasoning_reward(completions, answer, question, **kwargs) -> list[float]:
    """
    Partial credit for correct intermediate computation steps.

    Uses GSM8K's <<expression=result>> annotations in gold labels.
    For each intermediate result that appears in the model's response
    (but NOT in the question), award partial credit.

    Scaled to [0.0, 0.5] so correctness_reward (1.0) always dominates.

    This provides gradient signal even when the final answer is wrong:
    a model that computed 3/4 intermediate steps correctly understood
    more of the problem than one that computed 0/4.
    """
    rewards = []
    debug_info = []

    for completion, gold_label, q in zip(completions, answer, question):
        text = _get_text(completion)

        # Step 1: Extract all intermediate computation results from gold label
        all_gold_steps = extract_gold_steps(gold_label)

        # Step 2: Need at least 2 steps (1 intermediate + 1 final) for this
        # reward to be meaningful. Single-step problems → no partial credit.
        if len(all_gold_steps) < 2:
            rewards.append(0.0)
            debug_info.append(("skip", 0, 0, []))
            continue

        # Step 3: Drop the LAST step -- it equals the final answer,
        # which is already rewarded by correctness_reward. No double counting.
        intermediate_steps = all_gold_steps[:-1]

        # Step 4: Get numbers from the question. We exclude intermediate
        # results that match question numbers because finding "430" in the
        # response when the question says "$430" doesn't prove computation.
        question_nums = extract_question_numbers(q)

        # Step 5: Keep only intermediate results NOT in the question
        unique_steps = []
        for step_val in intermediate_steps:
            # Check if this value appears in the question
            in_question = any(abs(step_val - qn) < 1e-6 for qn in question_nums)
            if not in_question:
                unique_steps.append(step_val)

        if not unique_steps:
            rewards.append(0.0)
            debug_info.append(("no_unique", len(intermediate_steps), 0, []))
            continue

        # Step 6: Extract all numbers from the model's response
        response_numbers = _extract_all_numbers(text)

        # Step 7: Check which unique intermediate results appear in response
        found_steps = []
        for step_val in unique_steps:
            found = any(abs(step_val - rn) < 1e-6 for rn in response_numbers)
            found_steps.append((step_val, found))

        n_found = sum(1 for _, f in found_steps if f)
        n_total = len(unique_steps)

        # Step 8: Score = fraction of unique intermediate steps found, scaled to [0, 0.5]
        score = (n_found / n_total) * 0.5
        rewards.append(score)
        debug_info.append(("scored", n_total, n_found, found_steps))

    # Debug logging (on same schedule as correctness)
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  REASONING (step {_step_counter}) — avg={avg_reward:.3f}", flush=True)
        for i, (status, n_total, n_found, steps) in enumerate(debug_info[:4]):
            if status == "scored":
                step_str = ", ".join(
                    f"{v:.0f}={'Y' if f else 'N'}" for v, f in steps
                )
                print(f"  [{i}] {n_found}/{n_total} steps found → "
                      f"reward={rewards[i]:.3f}  [{step_str}]", flush=True)
            else:
                print(f"  [{i}] {status} → reward=0.0", flush=True)
        print(flush=True)

    return rewards


# ── Reward function 3: Format (structured reasoning) ────────────────────────

def format_reward(completions, **kwargs) -> list[float]:
    """
    Reward structured step-by-step reasoning. Scaled to [0.0, 0.2].

    Checks for evidence of organized problem-solving:
      - Numbered steps or transitional phrases ("Step 1:", "First,", "1.")
      - A clearly marked final answer (\\boxed{}, ####, "the answer is")
      - Penalizes responses that jump straight to an answer with no work

    This targets "early termination" and "format failure" error modes
    identified in gap analysis.
    """
    rewards = []
    debug_info = []

    for completion in completions:
        text = _get_text(completion)
        score = 0.0
        reasons = []

        # ── Check 1: Step-by-step structure (0.0 - 0.10) ───────────────
        # Look for numbered steps, bullet points, or transitional phrases
        step_patterns = [
            r"(?:Step|step)\s+\d",             # "Step 1", "step 2"
            r"^\s*\d+\.\s",                    # "1. ", "2. " at line start
            r"(?:First|Second|Third|Next|Then|Finally)[,:]",  # transitions
            r"^\s*[-*]\s",                     # bullet points
        ]
        n_step_markers = 0
        for pattern in step_patterns:
            n_step_markers += len(re.findall(pattern, text, re.MULTILINE))

        if n_step_markers >= 3:
            score += 0.10
            reasons.append(f"steps={n_step_markers}")
        elif n_step_markers >= 1:
            score += 0.05
            reasons.append(f"steps={n_step_markers}")

        # ── Check 2: Clear final answer format (0.0 - 0.05) ────────────
        has_boxed = bool(re.search(r"\\boxed\{", text))
        has_hash = bool(re.search(r"####", text))
        has_answer_phrase = bool(re.search(
            r"[Tt]he\s+(?:final\s+)?answer\s+is", text
        ))
        if has_boxed or has_hash or has_answer_phrase:
            score += 0.05
            fmt = "boxed" if has_boxed else ("####" if has_hash else "phrase")
            reasons.append(f"answer_fmt={fmt}")

        # ── Check 3: Not too short (penalize bare answers) (0.0 - 0.05)
        word_count = len(text.split())
        if word_count >= 30:
            score += 0.05
            reasons.append(f"words={word_count}")
        elif word_count < 10:
            # Bare answer with no reasoning at all → no format reward
            score = 0.0
            reasons = ["too_short"]

        rewards.append(round(score, 3))
        debug_info.append(reasons)

    # Debug logging
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  FORMAT (step {_step_counter}) — avg={avg_reward:.3f}", flush=True)
        for i, reasons in enumerate(debug_info[:4]):
            print(f"  [{i}] reward={rewards[i]:.3f}  [{', '.join(reasons)}]",
                  flush=True)
        print(flush=True)

    return rewards


# ── Reward function 4: Hallucination penalty (number grounding) ──────────────

def hallucination_penalty(completions, answer, question, **kwargs) -> list[float]:
    """
    Penalize introducing numbers that appear neither in the question nor in
    any valid intermediate computation step. Scaled to [-0.3, 0.0].

    This targets "premise mismatch" errors where the model invents quantities
    (e.g., using "$500" when the question says "$430").

    Numbers <= 10 are ignored as they commonly appear as counters, ordinals,
    or small multipliers that don't indicate hallucination.
    """
    rewards = []
    debug_info = []

    for completion, gold_label, q in zip(completions, answer, question):
        text = _get_text(completion)

        # Build the set of "grounded" numbers: question + gold intermediate steps
        question_nums = extract_question_numbers(q)
        gold_steps = extract_gold_steps(gold_label)
        grounded_nums = question_nums | set(gold_steps)

        # Extract all numbers from the model's response
        response_nums = _extract_all_numbers(text)

        # Filter to "suspicious" numbers: > 10 and not grounded
        ungrounded = set()
        for rn in response_nums:
            if rn <= 10:
                continue  # skip small numbers (counters, ordinals)
            is_grounded = any(abs(rn - gn) < 1e-6 for gn in grounded_nums)
            if not is_grounded:
                ungrounded.add(rn)

        # Penalty: proportional to fraction of response numbers that are ungrounded
        # Only count non-trivial numbers (> 10) in the denominator
        nontrivial_response = {rn for rn in response_nums if rn > 10}
        if not nontrivial_response:
            penalty = 0.0
        else:
            hallucination_ratio = len(ungrounded) / len(nontrivial_response)
            # Scale: 0% ungrounded → 0.0, 100% ungrounded → -0.3
            penalty = -0.3 * hallucination_ratio

        rewards.append(round(penalty, 3))
        debug_info.append((len(ungrounded), len(nontrivial_response),
                          sorted(ungrounded)[:5]))

    # Debug logging
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  HALLUCINATION (step {_step_counter}) — avg={avg_reward:.3f}",
              flush=True)
        for i, (n_ungrounded, n_total, examples) in enumerate(debug_info[:4]):
            print(f"  [{i}] {n_ungrounded}/{n_total} ungrounded → "
                  f"penalty={rewards[i]:.3f}  examples={examples}",
                  flush=True)
        print(flush=True)

    return rewards


# ── Reward function 5: Repetition penalty ────────────────────────────────────

def repetition_penalty(completions, **kwargs) -> list[float]:
    """
    Penalize responses with repeated reasoning patterns. Scaled to [-0.2, 0.0].

    Detects two types of repetition:
      1. N-gram repetition: trigrams appearing 3+ times (reasoning loop)
      2. Line repetition: exact duplicate lines (copy-paste loop)

    This targets "repetition/logical loop" errors from gap analysis.
    """
    rewards = []
    debug_info = []

    for completion in completions:
        text = _get_text(completion)
        penalty = 0.0

        # ── Check 1: Trigram repetition ratio ───────────────────────────
        words = text.lower().split()
        trigram_ratio = 0.0
        if len(words) >= 5:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            counts = Counter(trigrams)
            repeated = sum(1 for t in trigrams if counts[t] >= 3)
            trigram_ratio = repeated / len(trigrams)

        # ── Check 2: Repeated lines ─────────────────────────────────────
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 15]
        line_counts = Counter(lines)
        max_line_repeat = max(line_counts.values()) if line_counts else 0

        # ── Combine into penalty ────────────────────────────────────────
        # Trigram repetition: ratio > 0.15 starts incurring penalty
        if trigram_ratio > 0.15:
            # Scale: 0.15 → 0.0, 0.60 → -0.15 (linear)
            trigram_penalty = -0.15 * min((trigram_ratio - 0.15) / 0.45, 1.0)
            penalty += trigram_penalty

        # Line repetition: any line repeated 3+ times incurs penalty
        if max_line_repeat >= 3:
            # Scale: 3 repeats → -0.05, 6+ repeats → -0.10
            line_penalty = -0.05 * min(max_line_repeat / 3, 2.0)
            penalty += line_penalty

        # Clamp to [-0.2, 0.0]
        penalty = max(penalty, -0.2)
        rewards.append(round(penalty, 3))
        debug_info.append((round(trigram_ratio, 3), max_line_repeat))

    # Debug logging
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  REPETITION (step {_step_counter}) — avg={avg_reward:.3f}",
              flush=True)
        for i, (tri_ratio, max_rep) in enumerate(debug_info[:4]):
            print(f"  [{i}] trigram_ratio={tri_ratio}, max_line_repeat={max_rep} → "
                  f"penalty={rewards[i]:.3f}", flush=True)
        print(flush=True)

    return rewards
