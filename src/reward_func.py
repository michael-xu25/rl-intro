"""
Rule-Based Reward Functions for Entity-Tracking GRPO on GSM8K.

Designed from Pass@16 analysis of Qwen2.5-1.5B-Instruct:
- pass@1: 67.7%, pass@16: 95%
- Key failure mode: entity tracking (forgetting people/items in multi-entity problems)
- Training dataset: GSM8K filtered to 3+ entity problems

Two reward functions:
  1. correctness_reward:      1.0 if final answer matches gold, 0.0 otherwise
  2. entity_tracking_reward:  0.0-0.5 partial credit for each entity accounted for
                              (name mentioned + number computed nearby in <think>)
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


# ── Entity tracking helpers ──────────────────────────────────────────────────

def _extract_think_section(text: str) -> str:
    """Extract content inside <think>...</think> tags, or fall back to full text."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _entity_has_computation(entity_name: str, text: str) -> bool:
    """Check if an entity name appears near a number in the text.

    Looks for the entity name followed by a number within ~200 characters.
    This checks that the model actually computed something for this entity,
    not just mentioned the name in passing.

    Examples that match:
        "Kim raises $320 + $430 = $750"     (Kim + number nearby)
        "Alexandra: $430"                    (Alexandra + number nearby)
        "Sarah, who raises $300"             (Sarah + number nearby)

    Examples that don't match:
        "We need to find the total for all"  (no entity name)
        "Kim is one of the girls"            (name but no number nearby)
    """
    # Search for entity name followed (within 200 chars) by a number
    # The number can be preceded by $, =, :, etc.
    pattern = re.escape(entity_name) + r"[\s\S]{0,200}?[\$=:\s][\d,]+(?:\.\d+)?"
    if re.search(pattern, text, re.IGNORECASE):
        return True

    # Also check reverse: number followed by entity name (less common but valid)
    # e.g., "$750 for Kim"
    pattern_rev = r"[\d,]+(?:\.\d+)?[\s\S]{0,50}?" + re.escape(entity_name)
    if re.search(pattern_rev, text, re.IGNORECASE):
        return True

    return False


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


# ── Reward function 2: Entity Tracking ──────────────────────────────────────

def entity_tracking_reward(completions, entities, **kwargs) -> list[float]:
    """
    Partial credit for each entity the model explicitly accounts for.

    For each entity name in the pre-extracted `entities` list (from the
    curated dataset), checks if the model's output:
    1. Mentions the entity name
    2. Has a number computed within ~200 characters of that mention

    Score = (entities with name + nearby number) / (total entities) * 0.5
    Max reward: 0.5 (correctness_reward at 1.0 always dominates)

    This provides gradient signal even when the final answer is wrong:
    a model that tracked 3/4 entities but made an arithmetic error
    gets 0.375, while one that only tracked 1/4 entities gets 0.125.
    """
    rewards = []
    debug_info = []

    for completion, entity_list in zip(completions, entities):
        text = _get_text(completion)

        # Parse entity_list -- may be a list or a JSON string
        if isinstance(entity_list, str):
            import json
            try:
                entity_list = json.loads(entity_list)
            except (json.JSONDecodeError, TypeError):
                entity_list = []

        # Need at least 2 entities for this reward to be meaningful
        if not entity_list or len(entity_list) < 2:
            rewards.append(0.0)
            debug_info.append(("skip", 0, 0, []))
            continue

        # Prefer <think> section, fall back to full text
        reasoning_text = _extract_think_section(text)

        # Check each entity
        tracked = []
        for entity_name in entity_list:
            found = _entity_has_computation(entity_name, reasoning_text)
            tracked.append((entity_name, found))

        n_found = sum(1 for _, f in tracked if f)
        n_total = len(entity_list)

        # Score = fraction of entities tracked, scaled to [0, 0.5]
        score = (n_found / n_total) * 0.5
        rewards.append(score)
        debug_info.append(("scored", n_total, n_found, tracked))

    # Debug logging (on same schedule as correctness)
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  ENTITY TRACKING (step {_step_counter}) — avg={avg_reward:.3f}", flush=True)
        for i, (status, n_total, n_found, tracked) in enumerate(debug_info[:4]):
            if status == "scored":
                entity_str = ", ".join(
                    f"{name}={'Y' if f else 'N'}" for name, f in tracked
                )
                print(f"  [{i}] {n_found}/{n_total} entities tracked → "
                      f"reward={rewards[i]:.3f}  [{entity_str}]", flush=True)
            else:
                print(f"  [{i}] {status} → reward=0.0", flush=True)
        print(flush=True)

    return rewards
