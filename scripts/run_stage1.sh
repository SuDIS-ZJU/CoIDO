#!/bin/bash
# COIDO: Efficient Data Selection for Visual Instruction Tuning
# Paper: https://arxiv.org/abs/XXXX.XXXXX
# Adapted from: https://github.com/haotian-liu/LLaVA

# Environment variables for training
export DS_SKIP_CUDA_CHECK=1
export NCCL_TIMEOUT=3600
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DISABLE_FLASH_ATTENTION=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT="COIDO"

# Paths - set these before running
: ${COIDO_DATA_DIR:="./data"}                    # Directory containing scores, clustering results
: ${COIDO_MODEL_CKPT:="/path/to/vicuna-7b-v1.5"}    # Vicuna checkpoint path
: ${COIDO_CLIP_CKPT:="/path/to/clip-vit-large-patch14"}  # CLIP checkpoint path
: ${COIDO_MM_ADAPTER:="/path/to/mm_projector.bin"}  # MM projector weights
: ${OUTPUT_DIR:="./checkpoints/coido_scorer"}    # Output directory

# Feature and clustering configuration
FEATURE_TYPE="clip+scores"
CLUSTERING_RESULTS_PATH="${COIDO_DATA_DIR}/gmm_clip+scores_20clusters.json"

# Model configuration
MODEL_VERSION="vicuna-v1-5-7b"

# Resume training from checkpoint if exists
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-*"
RESUME_TRAINING=""
if ls $CHECKPOINT_DIR 1> /dev/null 2>&1; then
    LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR | sort -V | tail -n 1)
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming training from checkpoint..."
    RESUME_TRAINING="--resume_from_checkpoint $LATEST_CHECKPOINT"
fi

mkdir -p $OUTPUT_DIR

deepspeed --include localhost:4,5,6,7 --master_port 12345 coido_scorer/stage1.py \
    --feature_extractor_setting $FEATURE_TYPE \
    --deepspeed ./LLaVA/scripts/zero3.json \
    --model_name_or_path ${COIDO_MODEL_CKPT} \
    --version v1_5 \
    --data_path ${COIDO_DATA_DIR}/llava_v1_5_665k_add_idx_with_dataset.json \
    --image_folder ./data \
    --vision_tower ${COIDO_CLIP_CKPT} \
    --pretrain_mm_mlp_adapter ${COIDO_MM_ADAPTER} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
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

echo "Training complete! COIDO Scorer saved to $OUTPUT_DIR"