"""
Rule-Based Reward Functions for GSM8K Math Problems.

Designed after analyzing 100 baseline eval samples of Qwen 0.5B Instruct:
- Model naturally uses \\boxed{} format 76% of the time
- 45% baseline accuracy
- Main failure mode: comprehension errors (80% of wrong answers)
- Parser must handle: \\boxed{}, last-number fallback, numeric comparison
"""

import re
import logging

logger = logging.getLogger("tiny_math_solver")

_step_counter = 0
LOG_EVERY = 10


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


def extract_gold_answer(label: str) -> str | None:
    """Extract the number after '####' in a GSM8K answer."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    """
    Extract the model's final answer. Returns (answer, method).
    Matches the model's natural behavior observed in baseline eval.
    """
    # 1. \boxed{<number>} -- model's preferred format (76% of responses)
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", ""), "boxed"

    # 2. #### <number> (GSM8K format, rare but possible after training)
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", ""), "####"

    # 3. "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", ""), "the_answer_is"

    # 4. **<number>** (bold)
    matches = re.findall(r"\*\*([\d,]+(?:\.\d+)?)\*\*", text)
    if matches:
        return matches[-1].replace(",", ""), "bold"

    # 5. Last number in text (fallback -- actually reliable per eval analysis)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number_fallback"

    return None, "no_answer"


def _numbers_match(pred: str, gold: str) -> bool:
    """Numeric comparison instead of string comparison. Handles '38.00' == '38'."""
    pred_f = _normalize_number(pred)
    gold_f = _normalize_number(gold)
    if pred_f is None or gold_f is None:
        return False
    return abs(pred_f - gold_f) < 1e-6


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """
    Core reward: 1.0 if numerically correct, 0.0 otherwise.
    Uses numeric comparison (not string) to avoid false negatives.
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

    # Log samples: every step for first 5, then every LOG_EVERY
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        n_correct = int(sum(rewards))
        n_parsed = sum(1 for _, _, pred, _, _ in samples if pred is not None)
        print(flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  REWARD DEBUG (step {_step_counter}) — "
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


def format_reward(completions, **kwargs) -> list[float]:
    """
    Reward clear answer formatting. Based on baseline eval:
    - \\boxed{} is the model's natural format (76% usage) → reward it
    - Other explicit formats get partial credit
    - No answer format → 0.0

    This encourages the model to clearly state answers (makes parsing reliable).
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        if re.search(r"\\boxed\{[\d,]+(?:\.\d+)?\}", text):
            rewards.append(0.5)   # natural format, reward it
        elif re.search(r"####\s*[\d,]+", text):
            rewards.append(0.5)
        elif re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*[\d,]+", text):
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards
