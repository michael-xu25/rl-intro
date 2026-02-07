# Tiny-Math-Solver

Train **Qwen/Qwen2.5-0.5B-Instruct** to solve grade-school math problems (GSM8K) using **GRPO** (Group Relative Policy Optimization) with a **rule-based reward function** — powered by [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

No trained reward model needed. The reward function simply checks whether the model's numerical answer matches the ground truth.

## Quick Start (4 steps)

### 1. Push to GitHub (local machine)

```bash
cd rl-intro
git add -A && git commit -m "initial scaffold" && git push -u origin main
```

### 2. Clone on Lightning AI Studio

Open a **Lightning AI Studio** with a GPU (A10G or A100), then:

```bash
cd ~
git clone https://github.com/michael-xu25/rl-intro.git
cd rl-intro
```

### 3. Run setup (once)

```bash
bash setup_lightning.sh
```

This installs OpenRLHF, flash-attn, and pre-downloads the model + dataset.

### 4. Run training

```bash
bash src/train_math.sh
```

Optionally enable Weights & Biases logging:

```bash
WANDB_TOKEN=your_token_here bash src/train_math.sh
```

## Project Structure

```
rl-intro/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_lightning.sh        # One-time Lightning AI setup
├── .gitignore
└── src/
    ├── train_math.sh         # Main training launch script
    └── reward_func.py        # Rule-based reward (answer checker)
```

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop (GRPO)                  │
│                                                         │
│  GSM8K prompt ──► Actor (Qwen 0.5B + LoRA)             │
│                      │                                  │
│                      ▼  (4 samples per prompt)          │
│                  Generated solutions                    │
│                      │                                  │
│                      ▼                                  │
│               reward_func.py                            │
│              ┌───────────────┐                          │
│              │ Extract pred  │ ◄── model response       │
│              │ Extract gold  │ ◄── GSM8K label          │
│              │ Compare nums  │                          │
│              └──────┬────────┘                          │
│                     │                                   │
│              reward: 1.0 (correct) + 0.5 (format)      │
│              reward: 0.0 (wrong)                        │
│                     │                                   │
│                     ▼                                   │
│            GRPO advantage estimation                    │
│            (group normalisation across 4 samples)       │
│                     │                                   │
│                     ▼                                   │
│             Update LoRA adapter                         │
└─────────────────────────────────────────────────────────┘
```

## Configuration

Key knobs in `src/train_math.sh`:

| Parameter | Default | Notes |
|---|---|---|
| `LORA_RANK` | 16 | Higher = more capacity, more memory |
| `N_SAMPLES` | 4 | Samples per prompt for GRPO (try 8 for better signal) |
| `MICRO_TRAIN_BS` | 2 | Increase to 4-8 on A100 80GB |
| `MAX_SAMPLES` | 50000 | Total training prompts |
| `LR` | 5e-5 | Actor learning rate |
| `KL_COEF` | 0.05 | KL penalty (lower = more exploration) |

## After Training

The training script saves a **LoRA adapter** (not full weights). To merge it with the base model:

```bash
python -m openrlhf.cli.lora_combiner \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --lora_path ./checkpoint/qwen-0.5b-gsm8k-grpo \
    --output_path ./checkpoint/qwen-0.5b-gsm8k-merged \
    --bf16
```

## Switching Algorithms

Change `--advantage_estimator` in `src/train_math.sh`:

- `group_norm` — **GRPO** (default, no critic needed)
- `reinforce_baseline` — **REINFORCE++-baseline** (recommended for RLVR)
- `reinforce` — **REINFORCE++**
- `gae` — **PPO** (needs critic, uses more memory)

## References

- [OpenRLHF Documentation](https://openrlhf.readthedocs.io/)
- [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
