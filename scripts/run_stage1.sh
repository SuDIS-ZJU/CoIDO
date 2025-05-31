#!/bin/bash
# Self-Filter Stage 1 Training Script
# Adopted from https://github.com/haotian-liu/LLaVA

# Environment configuration
export NCCL_TIMEOUT=3600
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export WANDB_MODE=offline

# Wandb configuration
export WANDB_PROJECT="Self-Filter"
export WANDB_ENTITY="your-wandb-username"

# Training configuration
N_CLUSTERS=20
CLUSTERING_RESULTS_PATH="./data/spectral_clustering_clip+scores_${N_CLUSTERS}_multi_gpu.json"

# Model version settings
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-5-7b"

# Output directory
OUTPUT_DIR="./data/checkpoints/self-filter_stage1_${N_CLUSTERS}CLS"

# Check for existing checkpoints for resume training
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-*"
RESUME_TRAINING=""

if ls $CHECKPOINT_DIR 1> /dev/null 2>&1; then
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR | sort -V | tail -n 1)
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming training from checkpoint..."
    RESUME_TRAINING="--resume_from_checkpoint $LATEST_CHECKPOINT"
else
    echo "No checkpoint found, starting training from scratch..."
fi

# Ensure output directory exists with proper permissions
mkdir -p $OUTPUT_DIR
chmod 755 $OUTPUT_DIR

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12345 ./coido/stage1.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./path-to-models/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/training_data.json \
    --image_folder ./data/images \
    --vision_tower ./path-to-models/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./path-to-models/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --group_by_modality_length True \
    --use_clustering \
    --clustering_results_path $CLUSTERING_RESULTS_PATH \
    $RESUME_TRAINING

echo "Stage 1 training completed! Model saved to $OUTPUT_DIR" 