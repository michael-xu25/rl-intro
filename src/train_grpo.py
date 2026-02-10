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
from reward_func import (
    correctness_reward,
    reasoning_reward,
    format_reward,
    hallucination_penalty,
    repetition_penalty,
)

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

    # GRPO sampling — based on pass@16 analysis of 1.5B model:
    # - pass@1=67.7%, pass@16=95%, 33 problems in RL sweet spot
    # - avg response ~430 tokens on hard problems, so 1024 gives full room
    # - 8 generations at temp=0.9 gives good mix of correct/wrong per group
    num_generations=8,
    max_completion_length=1024,
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


# ── Model selection (supports SFT warm-up checkpoint) ──────────────────────
# If SFT_CHECKPOINT env var is set, start GRPO from that checkpoint instead
# of the raw base model. This is the "teach the model to be its best self"
# step from the hybrid training approach.
SFT_CHECKPOINT = os.environ.get("SFT_CHECKPOINT", None)
MODEL_NAME = SFT_CHECKPOINT if SFT_CHECKPOINT else "Qwen/Qwen2.5-1.5B-Instruct"


# ── Log config ──────────────────────────────────────────────────────────────
logger.info("Config:")
logger.info(f"  Model:            {MODEL_NAME}")
if SFT_CHECKPOINT:
    logger.info(f"  SFT checkpoint:   {SFT_CHECKPOINT}")
logger.info(f"  LoRA rank:        {peft_config.r}")
logger.info(f"  Num generations:  {training_args.num_generations}")
logger.info(f"  Batch size:       {training_args.per_device_train_batch_size}")
logger.info(f"  Grad accum steps: {training_args.gradient_accumulation_steps}")
logger.info(f"  Learning rate:    {training_args.learning_rate}")
logger.info(f"  Temperature:      {training_args.temperature}")
logger.info(f"  Output dir:       {training_args.output_dir}")
logger.info(f"  W&B:              {training_args.report_to}")


# ── Trainer ─────────────────────────────────────────────────────────────────
# Five thematic reward functions (from gap analysis):
#   correctness_reward:    0.0 or 1.0  — final answer match (dominant signal)
#   reasoning_reward:      0.0 - 0.5   — intermediate step verification
#   format_reward:         0.0 - 0.2   — structured step-by-step reasoning
#   hallucination_penalty: -0.3 - 0.0  — penalize ungrounded numbers
#   repetition_penalty:    -0.2 - 0.0  — penalize reasoning loops
# Total range: -0.5 to 1.7, correctness always dominates.
trainer = GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        correctness_reward,
        reasoning_reward,
        format_reward,
        hallucination_penalty,
        repetition_penalty,
    ],
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
