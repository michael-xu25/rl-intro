"""
Rule-Based Reward Function for GSM8K Math Problems.

This module is consumed by OpenRLHF's Reinforced Fine-tuning feature.
Pass its path to --remote_rm_url in the training script.

Signature required by OpenRLHF:
    reward_func(queries, prompts, labels) -> dict

- queries:  list[str] — full text (prompt + model response)
- prompts:  list[str] — the original prompt only
- labels:   list[str] — the ground-truth answer field from the dataset
                         (populated via --label_key in the training script)

Returns:
    dict with keys "rewards", "scores", and "extra_logs"
"""

import re
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_gold_answer(label: str) -> str | None:
    """
    Extract the reference number from a GSM8K label.

    GSM8K answers look like:
        "...reasoning steps...\\n#### 42"
    We grab the number after '####'.
    """
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    if match:
        # Remove commas (e.g. "1,200" -> "1200")
        return match.group(1).replace(",", "")
    return None


def extract_predicted_answer(response: str) -> str | None:
    """
    Extract the predicted number from the model's generated response.

    Strategy (in priority order):
      1. Look for '#### <number>' (model learned the GSM8K format)
      2. Look for 'The answer is <number>'
      3. Fall back to the last number in the response
    """
    # Strategy 1: #### pattern
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", response)
    if match:
        return match.group(1).replace(",", "")

    # Strategy 2: "the answer is <number>"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*([\d,]+(?:\.\d+)?)", response
    )
    if match:
        return match.group(1).replace(",", "")

    # Strategy 3: last number in text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


# ---------------------------------------------------------------------------
# Main reward function (entry point for OpenRLHF)
# ---------------------------------------------------------------------------

def reward_func(queries, prompts, labels):
    """
    Compute rewards for a batch of (query, prompt, label) triples.

    Returns dict:
        rewards     — tensor of floats used for advantage calculation
        scores      — tensor of floats used for dynamic filtering (keep in 0-1)
        extra_logs  — dict of additional metrics logged to wandb
    """
    rewards = []
    correct_count = 0
    format_count = 0

    for query, prompt, label in zip(queries, prompts, labels):
        # The model response is everything after the prompt
        response = query[len(prompt):]

        gold = extract_gold_answer(label)
        pred = extract_predicted_answer(response)

        # --- Correctness reward (0 or 1) ---
        if gold is not None and pred is not None and pred == gold:
            reward = 1.0
            correct_count += 1
        else:
            reward = 0.0

        # --- Format bonus (+0.5 if model uses #### format) ---
        if re.search(r"####\s*[\d,]+", response):
            reward += 0.5
            format_count += 1

        rewards.append(reward)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # Normalise scores to 0-1 range for dynamic filtering
    max_possible = 1.5  # correctness (1.0) + format (0.5)
    scores = reward_tensor / max_possible

    return {
        "rewards": reward_tensor,
        "scores": scores,
        "extra_logs": {
            "accuracy": correct_count / max(len(queries), 1),
            "format_adherence": format_count / max(len(queries), 1),
        },
    }
