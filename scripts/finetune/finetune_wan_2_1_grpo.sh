# export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

# mkdir videos


# sudo apt-get update
# yes | sudo apt-get install python3-tk

# git clone https://github.com/tgxs002/HPSv2.git
# cd HPSv2
# pip install -e . 
# cd ..

LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node=4 --master_port 19002 \
    fastvideo/train_grpo_wan_2_1.py \
    --seed 42 \
    --pretrained_model_name_or_path /share/models/Wan2.1-T2V-1.3B-Diffusers \
    --vae_model_path /share/models/Wan2.1-T2V-1.3B-Diffusers \
    --cache_dir data/.cache \
    --data_json_path /share/models/Wan2.1-T2V-1.3B-Diffusers/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 24 \
    --max_train_steps 1000 \
    --learning_rate 2e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 1000 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo \
    --h 512 \
    --w 512 \
    --t 1 \
    --sampling_steps 20 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --use_hpsv2 \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 0.1 \
    --adv_clip_max 5.0 \
    --cfg_infer 5.0 2>&1 | tee "$LOG_FILE"