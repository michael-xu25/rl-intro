#!/usr/bin/env bash
# ============================================================================
# Tiny-Math-Solver — GRPO training on GSM8K with OpenRLHF
# ============================================================================
# Usage:   bash src/train_math.sh
# Assumes: single-GPU Lightning AI Studio (L4 24 GB or A100 40/80 GB)
# ============================================================================
set -euo pipefail

# ── Configurable knobs ──────────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET="openai/gsm8k"
SAVE_PATH="./checkpoint/qwen-0.5b-gsm8k-grpo"
REWARD_FUNC="$(pwd)/src/reward_func.py"           # rule-based reward
WANDB_TOKEN="${WANDB_TOKEN:-}"                      # set env var or leave blank

# LoRA
LORA_RANK=16
LORA_DROPOUT=0.05

# Batch sizes  (conservative for L4 24 GB — increase for A100)
MICRO_TRAIN_BS=2          # per-GPU training micro-batch
TRAIN_BS=64               # global training batch
MICRO_ROLLOUT_BS=4        # per-GPU rollout micro-batch
ROLLOUT_BS=64             # rollout buffer size
N_SAMPLES=4               # samples per prompt (GRPO needs >1)

# Generation
PROMPT_MAX_LEN=512
GENERATE_MAX_LEN=512

# Training schedule
MAX_SAMPLES=50000         # total prompts to train on
MAX_EPOCHS=1
LR=5e-5
KL_COEF=0.05

# ── Start Ray (single-node, single-GPU) ────────────────────────────────────
echo ">>> Starting Ray head (1 GPU) ..."
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

# Wait for Ray to register resources
echo ">>> Waiting for Ray to be ready ..."
for i in $(seq 1 30); do
    if ray status 2>&1 | grep -q "1.0 GPU"; then
        echo "    Ray is ready with 1 GPU (${i}s)."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "    WARNING: Timed out waiting for Ray GPU."
        ray status
    fi
    sleep 2
done

# ── Build wandb flag ────────────────────────────────────────────────────────
WANDB_FLAG=""
if [ -n "$WANDB_TOKEN" ]; then
    WANDB_FLAG="--use_wandb $WANDB_TOKEN"
fi

# ── Launch training ─────────────────────────────────────────────────────────
# Run directly with python (not ray job submit) to avoid dashboard agent issues.
# Ray is already started above, so the training script connects to it automatically.
echo ">>> Launching GRPO training ..."
python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --colocate_actor_ref \
    --vllm_num_engines 0 \
    --pretrain "$MODEL" \
    --remote_rm_url "$REWARD_FUNC" \
    --save_path "$SAVE_PATH" \
    --advantage_estimator group_norm \
    --use_kl_loss \
    --init_kl_coef "$KL_COEF" \
    --n_samples_per_prompt "$N_SAMPLES" \
    --micro_train_batch_size "$MICRO_TRAIN_BS" \
    --train_batch_size "$TRAIN_BS" \
    --micro_rollout_batch_size "$MICRO_ROLLOUT_BS" \
    --rollout_batch_size "$ROLLOUT_BS" \
    --max_samples "$MAX_SAMPLES" \
    --max_epochs "$MAX_EPOCHS" \
    --prompt_max_len "$PROMPT_MAX_LEN" \
    --generate_max_len "$GENERATE_MAX_LEN" \
    --prompt_data "$DATASET" \
    --input_key question \
    --label_key answer \
    --apply_chat_template \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate "$LR" \
    --lr_scheduler cosine \
    --gradient_checkpointing \
    --adam_offload \
    --attn_implementation flash_attention_2 \
    --lora_rank "$LORA_RANK" \
    --lora_dropout "$LORA_DROPOUT" \
    --packing_samples \
    --save_steps 50 \
    --logging_steps 1 \
    --temperature 0.7 \
    $WANDB_FLAG

echo ""
echo ">>> Training complete!  LoRA adapter saved to: $SAVE_PATH"
echo ">>> To merge LoRA adapter with base model, run:"
echo "    python -m openrlhf.cli.lora_combiner \\"
echo "        --model_path $MODEL \\"
echo "        --lora_path $SAVE_PATH \\"
echo "        --output_path ./checkpoint/qwen-0.5b-gsm8k-merged \\"
echo "        --bf16"
