"""
Evaluate a GRPO-trained LoRA checkpoint on GSM8K.

Same methodology as eval_pass_at_k.py (same 100 test questions, same seed,
same answer extraction) so results are directly comparable to the 67.7%
baseline pass@1.

Usage:
    # Auto-find latest checkpoint:
    python src/eval_checkpoint.py

    # Specify checkpoint path:
    python src/eval_checkpoint.py --checkpoint checkpoint/run_20260215/checkpoint-300

    # Compare with and without the training system prompt:
    python src/eval_checkpoint.py --no-system-prompt
"""

import argparse
import glob
import json
import os
import re
import random
import time
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── Config (MUST match eval_pass_at_k.py for fair comparison) ────────────────
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_PROBLEMS = 100
K = 16
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 1024
SEED = 42

# The system prompt used during GRPO training
TRAINING_SYSTEM_PROMPT = (
    "Think step by step inside <think> tags before answering. "
    "For each person or item in the problem, explicitly state "
    "what you know about them and calculate their value."
)

# Baseline numbers from eval_pass_at_k.py (pre-training)
BASELINE_PASS_AT_1 = 0.677
BASELINE_PASS_AT_K = 0.95

random.seed(SEED)


# ── Answer extraction (identical to eval_pass_at_k.py) ───────────────────────

def extract_gold_answer(label: str) -> str | None:
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return match.group(1).replace(",", ""), "boxed"
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", ""), "####"
    match = re.search(
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text
    )
    if match:
        return match.group(1).replace(",", ""), "the_answer_is"
    matches = re.findall(r"\*\*([\d,]+(?:\.\d+)?)\*\*", text)
    if matches:
        return matches[-1].replace(",", ""), "bold"
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        match = re.fullmatch(r"\$?\s*([\d,]+(?:\.\d+)?)\s*\$?\s*\.?", line)
        if match:
            return match.group(1).replace(",", ""), "last_line_number"
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ""), "last_number_fallback"
    return None, "no_answer"


def nums_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return False


# ── Find latest checkpoint ───────────────────────────────────────────────────

def find_latest_checkpoint() -> str | None:
    """Find the most recent checkpoint directory."""
    patterns = [
        "checkpoint/run_*/checkpoint-*",
        "checkpoint/run_*",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    # Filter to directories that contain adapter_config.json (LoRA checkpoint)
    valid = [c for c in candidates if os.path.isfile(os.path.join(c, "adapter_config.json"))]
    if not valid:
        return None
    return max(valid, key=os.path.getmtime)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoint on GSM8K")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint (auto-detects if not set)")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Evaluate WITHOUT training system prompt (for baseline comparison)")
    args = parser.parse_args()

    # Find checkpoint
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    if checkpoint_path is None:
        print("ERROR: No checkpoint found. Specify --checkpoint or run training first.")
        return
    print(f"Checkpoint: {checkpoint_path}")

    use_system_prompt = not args.no_system_prompt
    print(f"System prompt: {'YES (training prompt)' if use_system_prompt else 'NO (baseline-compatible)'}")

    # Load model
    print(f"\nLoading {BASE_MODEL} + LoRA from {checkpoint_path} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Load same test questions
    print("Loading GSM8K (test split) ...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = random.sample(range(len(dataset)), min(N_PROBLEMS, len(dataset)))
    print(f"Selected {len(indices)} questions (same seed={SEED} as baseline)\n")

    # Run eval
    tag = "with_system_prompt" if use_system_prompt else "no_system_prompt"
    jsonl_path = f"logs/eval_checkpoint_{tag}.jsonl"
    os.makedirs("logs", exist_ok=True)

    print(f"{'='*70}")
    print(f"  CHECKPOINT EVAL: {checkpoint_path}")
    print(f"  {N_PROBLEMS} problems × {K} samples @ temp={TEMPERATURE}")
    print(f"  Baseline pass@1: {BASELINE_PASS_AT_1*100:.1f}%")
    print(f"{'='*70}\n")

    all_results = []
    total_correct_at_1 = 0
    total_pass_at_k = 0

    with open(jsonl_path, "w") as f:
        for prob_i, idx in enumerate(indices):
            question = dataset[idx]["question"]
            gold_label = dataset[idx]["answer"]
            gold = extract_gold_answer(gold_label)

            # Build prompt (with or without system prompt)
            if use_system_prompt:
                messages = [
                    {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
            else:
                messages = [{"role": "user", "content": question}]

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            # Generate K solutions
            t_gen = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            gen_time = time.time() - t_gen

            # Score
            prompt_len = inputs["input_ids"].shape[1]
            n_correct = 0
            methods = Counter()
            for seq_i in range(K):
                gen_tokens = outputs[seq_i][prompt_len:]
                response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                pred, method = extract_predicted_answer(response)
                methods[method] += 1
                if nums_match(pred, gold):
                    n_correct += 1

            pass_rate = n_correct / K
            total_correct_at_1 += pass_rate
            if n_correct > 0:
                total_pass_at_k += 1

            record = {
                "idx": idx, "gold": gold, "n_correct": n_correct,
                "n_total": K, "pass_rate": round(pass_rate, 4),
                "methods": dict(methods),
            }
            all_results.append(record)
            f.write(json.dumps(record) + "\n")

            bar = "█" * n_correct + "░" * (K - n_correct)
            emoji = "✓" if n_correct > 0 else "✗"
            print(f"[{prob_i+1:3d}/{N_PROBLEMS}] {emoji} {n_correct:2d}/{K} correct  "
                  f"|{bar}|  gold={gold}  ({gen_time:.1f}s)")

    # Summary
    est_pass_at_1 = total_correct_at_1 / N_PROBLEMS
    pass_at_k = total_pass_at_k / N_PROBLEMS
    delta_1 = est_pass_at_1 - BASELINE_PASS_AT_1
    delta_k = pass_at_k - BASELINE_PASS_AT_K

    print(f"\n{'='*70}")
    print(f"  RESULTS: {checkpoint_path}")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Baseline':>10s} {'Checkpoint':>12s} {'Delta':>10s}")
    print(f"  {'-'*57}")
    print(f"  {'Estimated pass@1':<25s} {BASELINE_PASS_AT_1*100:>9.1f}% {est_pass_at_1*100:>11.1f}% {delta_1*100:>+9.1f}%")
    print(f"  {'Pass@16':<25s} {BASELINE_PASS_AT_K*100:>9.1f}% {pass_at_k*100:>11.1f}% {delta_k*100:>+9.1f}%")
    print(f"{'='*70}")

    # Difficulty histogram
    print(f"\n  --- Difficulty Distribution ---")
    buckets = {"0/16 (hopeless)": 0, "1-2/16 (hard)": 0, "3-5/16 (medium)": 0,
               "6-10/16 (learnable)": 0, "11-16/16 (easy)": 0}
    for r in all_results:
        nc = r["n_correct"]
        if nc == 0: buckets["0/16 (hopeless)"] += 1
        elif nc <= 2: buckets["1-2/16 (hard)"] += 1
        elif nc <= 5: buckets["3-5/16 (medium)"] += 1
        elif nc <= 10: buckets["6-10/16 (learnable)"] += 1
        else: buckets["11-16/16 (easy)"] += 1

    for bucket, count in buckets.items():
        bar = "█" * count
        print(f"  {bucket:25s}: {count:3d}  {bar}")

    # Verdict
    print()
    if delta_1 > 0.03:
        print(f"  ✓ IMPROVEMENT: pass@1 up {delta_1*100:+.1f}% from baseline")
    elif delta_1 > -0.03:
        print(f"  ~ NO CHANGE: pass@1 within ±3% of baseline ({delta_1*100:+.1f}%)")
    else:
        print(f"  ✗ REGRESSION: pass@1 down {delta_1*100:+.1f}% from baseline")

    print(f"\n  Results saved to: {jsonl_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
