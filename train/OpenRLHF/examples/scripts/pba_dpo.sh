set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/pba-dpo \
   --save_steps -1 \
   --logging_steps 10 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --pretrain /meta-llama/Llama-3.1-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset /train.json \
   --prompt_key pba_prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi