"""
Evaluate the base Qwen 0.5B Instruct model on GSM8K before any RL training.

Prints sample outputs and calculates baseline accuracy, so we know what
GRPO needs to improve on.

Usage:
    python src/eval_baseline.py
"""

import re
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_SAMPLES = 50       # evaluate on 50 random questions
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
SEED = 42

random.seed(SEED)


# ── Answer extraction (same logic as reward_func.py) ────────────────────────
def extract_gold_answer(label: str) -> str | None:
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", label)
    return match.group(1).replace(",", "") if match else None

def extract_predicted_answer(text: str) -> str | None:
    # #### pattern
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    # "the answer is"
    match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*\$?\s*([\d,]+(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    # \boxed{}
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match: return match.group(1).replace(",", "")
    # **bold** at end
    match = re.search(r"\*\*([\d,]+(?:\.\d+)?)\*\*\s*$", text.strip())
    if match: return match.group(1).replace(",", "")
    return None


# ── Load model and data ─────────────────────────────────────────────────────
print(f"Loading {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

print("Loading GSM8K ...")
dataset = load_dataset("openai/gsm8k", "main", split="train")
indices = random.sample(range(len(dataset)), N_SAMPLES)


# ── Evaluate ────────────────────────────────────────────────────────────────
correct = 0
parsed = 0
total = 0

print(f"\n{'='*70}")
print(f"  BASELINE EVALUATION: {MODEL} on {N_SAMPLES} GSM8K questions")
print(f"{'='*70}\n")

for i, idx in enumerate(indices):
    question = dataset[idx]["question"]
    gold_label = dataset[idx]["answer"]
    gold = extract_gold_answer(gold_label)

    # Format as chat
    messages = [{"role": "user", "content": question}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Decode only the generated part
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_predicted_answer(response)

    is_correct = gold is not None and pred is not None and pred == gold
    if pred is not None:
        parsed += 1
    if is_correct:
        correct += 1
    total += 1

    # Print first 10 samples in detail, then just a summary line
    if i < 10:
        status = "CORRECT" if is_correct else "WRONG  "
        snippet = response[:300].replace('\n', ' | ')
        if len(response) > 300:
            snippet += "..."
        print(f"[{i+1:2d}/{N_SAMPLES}] [{status}] gold={gold}  pred={pred}")
        print(f"  Q: {question[:100]}...")
        print(f"  A: {snippet}")
        print()
    elif i % 10 == 0:
        print(f"[{i+1:2d}/{N_SAMPLES}] Running... ({correct}/{total} correct so far)")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  BASELINE RESULTS")
print(f"{'='*70}")
print(f"  Model:              {MODEL}")
print(f"  Questions evaluated: {total}")
print(f"  Answers parsed:     {parsed}/{total} ({100*parsed/total:.1f}%)")
print(f"  Correct:            {correct}/{total} ({100*correct/total:.1f}%)")
print(f"{'='*70}")
print()
print("This is your baseline. After GRPO training, run this again with the")
print("trained model to see if accuracy improved.")
