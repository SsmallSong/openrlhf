set -x 

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
    --reward_pretrain OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
    --save_path /root/.cache/huggingface/hub/7b_llama3_inst_ppo_openrlhf \
    --save_steps 20 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 256 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data princeton-nlp/llama3-ultrafeedback \
    --prompt_data_probs 1.0 \
    --max_samples 162000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --apply_chat_template \
    --input_key prompt
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
