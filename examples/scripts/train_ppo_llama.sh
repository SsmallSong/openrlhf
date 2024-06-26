set -x 

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain /root/.cache/huggingface/hub/llama2_sft_mirror/ \
    --reward_pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --save_path /root/.cache/huggingface/hub/7b_llama_ppo_openrlhf \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 512 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data trl-internal-testing/hh-rlhf-trl-style \
    --prompt_data_probs 1.0 \
    --max_samples 10 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
