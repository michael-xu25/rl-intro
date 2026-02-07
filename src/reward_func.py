"""
Rule-Based Reward Functions for GSM8K Math Problems.

Used by TRL's GRPOTrainer via the reward_funcs parameter.

TRL reward function signature:
    reward_func(completions, **kwargs) -> list[float]

When using chat templates, `completions` is a list of conversation lists,
e.g. [[{"role": "assistant", "content": "..."}]]. We extract the text
from the last assistant message.

Additional dataset columns (like 'answer') are passed as kwargs.
"""

import re
import logging

logger = logging.getLogger("tiny_math_solver")

# Track how many times correctness_reward is called (= training steps)
_step_counter = 0
LOG_EVERY = 10  # log sample outputs every N steps


def _get_text(completion) -> str:
    """Extract plain text from a completion (handles both str and chat format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Chat format: list of message dicts, grab last assistant content
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        return ""
    return str(completion)


def extract_gold_answer(label: str) -> str | None:
    """Extract the number after '####' in a GSM8K answer."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_predicted_answer(text: str) -> str | None:
    """
    Extract the predicted number from model output.

    Only matches explicit answer patterns -- no "last number" fallback,
    because that picks up random intermediate numbers and creates noise.

    Priority:
      1. '#### <number>' (GSM8K format)
      2. 'The answer is <number>'
      3. '\\boxed{<number>}' (common math format)
    """
    # Strategy 1: #### pattern
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Strategy 2: "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", "")

    # Strategy 3: \boxed{<number>}
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", "")

    # Strategy 4: bold or highlighted answer at end: **<number>**
    match = re.search(r"\*\*([\d,]+(?:\.\d+)?)\*\*\s*$", text.strip())
    if match:
        return match.group(1).replace(",", "")

    return None


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if the model's answer matches the gold answer, else 0.0.

    Args:
        completions: Model-generated responses.
        answer: Gold answers from GSM8K (format: "reasoning...\n#### 42").
    """
    global _step_counter
    _step_counter += 1

    rewards = []
    samples = []  # collect for logging
    for completion, gold_label in zip(completions, answer):
        text = _get_text(completion)
        gold = extract_gold_answer(gold_label)
        pred = extract_predicted_answer(text)
        correct = gold is not None and pred is not None and pred == gold
        rewards.append(1.0 if correct else 0.0)
        samples.append((text, gold, pred, correct))

    # Log samples: every step for first 5, then every LOG_EVERY steps
    should_log = _step_counter <= 5 or _step_counter % LOG_EVERY == 0
    if should_log:
        n_correct = sum(r for r in rewards)
        n_parsed = sum(1 for _, _, pred, _ in samples if pred is not None)
        print(flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  REWARD DEBUG (step {_step_counter}) — "
              f"{int(n_correct)}/{len(rewards)} correct, "
              f"{n_parsed}/{len(rewards)} had parseable answer", flush=True)
        print(f"{'='*70}", flush=True)
        # Show up to 4 samples
        for i, (text, gold, pred, correct) in enumerate(samples[:4]):
            snippet = text[:400].replace('\n', ' ↵ ')
            if len(text) > 400:
                snippet += "..."
            status = "CORRECT" if correct else "WRONG  "
            print(f"  [{status}] gold={gold}  pred={pred}", flush=True)
            print(f"  Output: {snippet}", flush=True)
            print(f"  ---", flush=True)
        print(f"{'='*70}", flush=True)
        print(flush=True)

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """
    Format reward: encourages the model to clearly state its answer.

    +0.5 for '#### <number>' format
    +0.3 for 'The answer is <number>'
    +0.1 for any clear answer pattern (boxed, bold)
    0.0 for no clear answer
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        if re.search(r"####\s*[\d,]+", text):
            rewards.append(0.5)
        elif re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*[\d,]+", text):
            rewards.append(0.3)
        elif re.search(r"\\boxed\{[\d,]+", text) or re.search(r"\*\*[\d,]+\*\*\s*$", text.strip()):
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards
