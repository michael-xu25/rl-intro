"""
Tiny-Math-Solver — GRPO training on GSM8K with TRL.

Usage:
    python src/train_grpo.py
    # or with accelerate for multi-GPU:
    accelerate launch src/train_grpo.py
"""

import os
import logging
from datetime import datetime, timezone

# ── Logging (set up BEFORE importing libraries that configure logging) ─────
RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{RUN_TIMESTAMP}.log")

# Explicitly add handlers to root logger (basicConfig is a no-op if
# any library has already configured logging, which transformers/trl do)
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.INFO)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_file_handler)

logger = logging.getLogger("tiny_math_solver")
logger.setLevel(logging.INFO)

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from reward_func import correctness_reward, format_reward

logger.info(f"=== Run started: {RUN_TIMESTAMP} ===")
logger.info(f"Log file: {LOG_FILE}")


# ── Dataset ─────────────────────────────────────────────────────────────────
# GSM8K has 'question' and 'answer' columns.
# TRL expects a 'prompt' column with chat messages.

def build_prompt(example):
    """Convert GSM8K question to chat format for Qwen Instruct.

    No system prompt -- baseline eval showed the model naturally uses
    \\boxed{} format 76% of the time and doesn't need format guidance.
    Let RL discover better reasoning, don't force a format.
    """
    example["prompt"] = [
        {"role": "user", "content": example["question"]},
    ]
    return example


logger.info("Loading GSM8K dataset ...")
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(build_prompt)
logger.info(f"Dataset loaded: {len(dataset)} examples")


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
    output_dir=f"./checkpoint/run_{RUN_TIMESTAMP}",
    run_name=f"grpo_{RUN_TIMESTAMP}",

    # GRPO sampling — based on baseline eval analysis:
    # - Model responses average 300 tokens (max 670), so 512 is safe
    # - 8 generations gives meaningful advantage signal
    # - temp=0.9 for diversity (some correct, some wrong → GRPO can learn)
    num_generations=8,
    max_completion_length=512,
    max_prompt_length=256,
    temperature=0.9,

    # Training — H200/A100 config
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16 prompts
    max_steps=200,                  # ~1hr on H200, extend later
    learning_rate=1e-4,             # moderate LR for LoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,

    # Logging & saving
    logging_steps=1,
    save_steps=50,
    report_to="wandb" if os.environ.get("WANDB_TOKEN") else "none",

    # Misc
    seed=42,
)


# ── Log config ──────────────────────────────────────────────────────────────
logger.info("Config:")
logger.info(f"  Model:            Qwen/Qwen2.5-0.5B-Instruct")
logger.info(f"  LoRA rank:        {peft_config.r}")
logger.info(f"  Num generations:  {training_args.num_generations}")
logger.info(f"  Batch size:       {training_args.per_device_train_batch_size}")
logger.info(f"  Grad accum steps: {training_args.gradient_accumulation_steps}")
logger.info(f"  Learning rate:    {training_args.learning_rate}")
logger.info(f"  Temperature:      {training_args.temperature}")
logger.info(f"  Output dir:       {training_args.output_dir}")
logger.info(f"  W&B:              {training_args.report_to}")


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

logger.info("Starting training ...")
trainer.train()
trainer.save_model()

logger.info(f"Training complete! Model saved to: {training_args.output_dir}")
logger.info(f"=== Run finished: {datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')} ===")
