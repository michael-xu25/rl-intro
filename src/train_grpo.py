"""
Tiny-Math-Solver — GRPO training on GSM8K with TRL.

Usage:
    python src/train_grpo.py
    # or with accelerate for multi-GPU:
    accelerate launch src/train_grpo.py
"""

import os
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from reward_func import correctness_reward, format_reward


# ── Dataset ─────────────────────────────────────────────────────────────────
# GSM8K has 'question' and 'answer' columns.
# TRL expects a 'prompt' column with chat messages.

def build_prompt(example):
    """Convert GSM8K question to chat format for Qwen Instruct."""
    example["prompt"] = [
        {"role": "user", "content": example["question"]},
    ]
    return example


dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(build_prompt)


# ── LoRA config ─────────────────────────────────────────────────────────────
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


# ── Training config ─────────────────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="./checkpoint/qwen-0.5b-gsm8k-grpo",

    # GRPO sampling
    num_generations=4,              # samples per prompt (G in the paper)
    max_completion_length=512,
    max_prompt_length=512,
    temperature=0.7,

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    gradient_checkpointing=True,

    # Logging & saving
    logging_steps=1,
    save_steps=50,
    report_to="wandb" if os.environ.get("WANDB_TOKEN") else "none",

    # Misc
    seed=42,
)


# ── Trainer ─────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[correctness_reward, format_reward],
    peft_config=peft_config,
)


# ── Train ───────────────────────────────────────────────────────────────────
if os.environ.get("WANDB_TOKEN"):
    import wandb
    wandb.login(key=os.environ["WANDB_TOKEN"])
    # If your W&B account requires a team entity, set WANDB_ENTITY env var
    if os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = os.environ["WANDB_ENTITY"]
        wandb.init(project="tiny-math-solver", entity=os.environ["WANDB_ENTITY"])

trainer.train()
trainer.save_model()

print(f"\nTraining complete! Model saved to: {training_args.output_dir}")
