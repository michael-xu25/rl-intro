"""
Rule-Based Reward Functions for GSM8K Math Problems.

Designed from Pass@16 analysis of Qwen2.5-1.5B-Instruct:
- pass@1: 67.7%, pass@16: 95%
- 33 problems in 1-10/16 range (RL sweet spot)
- Dominant error: comprehension (misreading which number modifies which noun)
- GSM8K gold labels contain <<expr=result>> intermediate step annotations

Two reward functions:
  1. correctness_reward: 1.0 if final answer matches gold, 0.0 otherwise
  2. reasoning_reward:   0.0-0.5 partial credit for correct intermediate steps
"""

import re
import logging

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
