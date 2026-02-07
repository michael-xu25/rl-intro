# Tiny-Math-Solver

Train **Qwen/Qwen2.5-0.5B-Instruct** to solve grade-school math problems (GSM8K) using **GRPO** with a **rule-based reward function** -- powered by [TRL](https://github.com/huggingface/trl).

No trained reward model needed. The reward function checks whether the model's numerical answer matches the ground truth.

## Quick Start

### 1. Clone on Lightning AI Studio

Open a **Lightning AI Studio** with a GPU (L4 recommended), then:

```bash
git clone https://github.com/michael-xu25/rl-intro.git
cd rl-intro
```

### 2. Run setup (once)

```bash
bash setup_lightning.sh
```

### 3. Run training

```bash
bash src/train_math.sh
```

With Weights & Biases logging:

```bash
WANDB_TOKEN=your_token bash src/train_math.sh
```

## Project Structure

```
rl-intro/
├── README.md
├── setup_lightning.sh          # One-time setup (pip install trl peft ...)
├── .gitignore
└── src/
    ├── train_math.sh           # Shell wrapper
    ├── train_grpo.py           # Training script (TRL GRPOTrainer)
    └── reward_func.py          # Rule-based rewards (answer checker)
```

## How It Works

```
GSM8K question
    │
    ▼
Actor (Qwen 0.5B + LoRA)
    │  generates 4 solutions per question
    ▼
reward_func.py
    ├── correctness_reward: 1.0 if answer matches gold, else 0.0
    └── format_reward:      0.5 if uses #### format, else 0.0
    │
    ▼
GRPO advantage = normalize rewards across the 4 samples
    │
    ▼
Update LoRA adapter
```

## Configuration

Edit `src/train_grpo.py` to tune:

| Parameter | Default | Notes |
|---|---|---|
| `r` (LoRA rank) | 16 | Higher = more capacity, more memory |
| `num_generations` | 4 | Samples per prompt (try 8 for better signal) |
| `per_device_train_batch_size` | 1 | Increase on A100 |
| `gradient_accumulation_steps` | 16 | Effective batch = this x per_device |
| `learning_rate` | 5e-5 | Actor learning rate |
| `temperature` | 0.7 | Generation temperature |

## After Training

The LoRA adapter is saved to `./checkpoint/qwen-0.5b-gsm8k-grpo`. To merge with base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "./checkpoint/qwen-0.5b-gsm8k-grpo")
merged = model.merge_and_unload()
merged.save_pretrained("./checkpoint/qwen-0.5b-gsm8k-merged")
```

## References

- [TRL GRPO Trainer Docs](https://huggingface.co/docs/trl/main/grpo_trainer)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [GRPO Paper (DeepSeekMath)](https://huggingface.co/papers/2402.03300)
