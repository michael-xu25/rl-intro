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

    Priority:
      1. '#### <number>' (GSM8K format)
      2. 'The answer is <number>'
      3. Last number in the text
    """
    # Strategy 1: #### pattern
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Strategy 2: "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", "")

    # Strategy 3: last number in text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """
    Binary reward: 1.0 if the model's answer matches the gold answer, else 0.0.

    Args:
        completions: Model-generated responses.
        answer: Gold answers from GSM8K (format: "reasoning...\n#### 42").
    """
    rewards = []
    for completion, gold_label in zip(completions, answer):
        text = _get_text(completion)
        gold = extract_gold_answer(gold_label)
        pred = extract_predicted_answer(text)
        if gold is not None and pred is not None and pred == gold:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Format reward: 0.5 if the model uses the '#### <number>' format, else 0.0.

    Encourages the model to produce answers in a parseable format.
    """
    rewards = []
    for completion in completions:
        text = _get_text(completion)
        if re.search(r"####\s*[\d,]+", text):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards
