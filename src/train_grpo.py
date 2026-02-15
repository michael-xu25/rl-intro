"""
Tiny-Math-Solver — GRPO on GSM8K with TRL.

Trains Qwen2.5-1.5B-Instruct on a curated subset of GSM8K problems
requiring 3+ entity tracking, using <think> tags for structured reasoning.

Reward functions:
  - correctness_reward (0/1): final answer matches gold
  - intermediate_step_reward (0-0.5): partial credit for correct intermediate
    computation values from gold <<expr=result>> annotations.

    This reward VARIES across completions (unlike the old entity_tracking_reward
    where all completions got identical scores), giving GRPO real gradients.

Usage:
    # First, build the curated dataset:
    python src/build_entity_dataset.py

    # Then train:
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

from datasets import load_from_disk
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from reward_func import correctness_reward, intermediate_step_reward

logger.info(f"=== Run started: {RUN_TIMESTAMP} ===")
logger.info(f"Log file: {LOG_FILE}")


# ── Dataset ─────────────────────────────────────────────────────────────────
# Curated dataset (built by src/build_entity_dataset.py).
# Contains GSM8K problems with 3+ named entities (the model's weak spot).
# The 'answer' column has gold solutions with <<expr=result>> annotations
# used by intermediate_step_reward. TRL expects a 'prompt' column.

ENTITY_TRACKING_PROMPT = (
    "Think step by step inside <think> tags before answering. "
    "For each person or item in the problem, explicitly state "
    "what you know about them and calculate their value."
)


def build_prompt(example):
    """Convert GSM8K question to chat format with <think> tag instruction.

    The system prompt forces structured entity-by-entity reasoning,
    creating diverse completions that GRPO can learn from.
    """
    example["prompt"] = [
        {"role": "system", "content": ENTITY_TRACKING_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    return example


logger.info("Loading entity-tracking dataset ...")
dataset = load_from_disk("data/entity_tracking_dataset")
dataset = dataset.map(build_prompt)
logger.info(f"Dataset loaded: {len(dataset)} entity-tracking problems")


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

    # GRPO sampling — entity-tracking config:
    # - Dataset filtered to 3+ entity problems (model's weak spot)
    # - <think> tags create structurally diverse reasoning paths
    # - temp=1.0 for maximum diversity across 8 generations
    # - 1024 completion length gives room for <think> section + answer
    # - 300 prompt length accommodates system prompt (~40 tokens)
    num_generations=8,
    max_completion_length=1024,
    max_prompt_length=300,
    temperature=1.0,

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
logger.info(f"  Model:            Qwen/Qwen2.5-1.5B-Instruct")
logger.info(f"  Mode:             GRPO (correctness + intermediate steps)")
logger.info(f"  System prompt:    {ENTITY_TRACKING_PROMPT[:60]}...")
logger.info(f"  LoRA rank:        {peft_config.r}")
logger.info(f"  Num generations:  {training_args.num_generations}")
logger.info(f"  Batch size:       {training_args.per_device_train_batch_size}")
logger.info(f"  Grad accum steps: {training_args.gradient_accumulation_steps}")
logger.info(f"  Learning rate:    {training_args.learning_rate}")
logger.info(f"  Temperature:      {training_args.temperature}")
logger.info(f"  Max prompt len:   {training_args.max_prompt_length}")
logger.info(f"  Max compl len:    {training_args.max_completion_length}")
logger.info(f"  Output dir:       {training_args.output_dir}")
logger.info(f"  W&B:              {training_args.report_to}")


# ── Trainer ─────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[correctness_reward, intermediate_step_reward],
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
