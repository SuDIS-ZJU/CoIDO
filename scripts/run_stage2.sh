#!/bin/bash
# COIDO Stage 2: Data Selection using trained COIDO Scorer
# Selects high-quality data subsets based on learned importance and diversity scores

# Paths - set these before running
: ${COIDO_DATA_DIR:="./data"}                    # Directory containing scores and clustering
: ${COIDO_STAGE1_CKPT:="./checkpoints/coido_scorer/checkpoint-1000"}  # Stage 1 checkpoint
: ${COIDO_OUTPUT_DIR:="./outputs"}              # Output directory for filtered dataset

STAGE1_MODEL_PATH="${COIDO_STAGE1_CKPT}"
FEATURE_TYPE="clip+scores"
RAW_ANNOTATION_PATH="${COIDO_DATA_DIR}/llava_v1_5_665k_add_idx_with_dataset.json"
RESULT_DIR="${COIDO_OUTPUT_DIR}/difficulty"
DATA_DIR="${COIDO_OUTPUT_DIR}/data"

DIFFICULTY_SAVE_NAME="difficulty_${FEATURE_TYPE}.json"
DIFFICULTY_SAVE_PATH="${RESULT_DIR}/${DIFFICULTY_SAVE_NAME}"
FILTERED_ANNOTATION_SAVE_PATH="${DATA_DIR}/llava_v1_5_filtered_dataset.json"
FILTER_NUM=133000
USE_FALLBACK=1

mkdir -p $RESULT_DIR
mkdir -p $DATA_DIR

echo "Running COIDO Stage 2: Data Selection..."
python coido_scorer/stage2.py \
    --stage1_model_path $STAGE1_MODEL_PATH \
    --feature_extractor_setting $FEATURE_TYPE \
    --result_dir $RESULT_DIR \
    --difficulty_save_name $DIFFICULTY_SAVE_NAME \
    --raw_annotation_path $RAW_ANNOTATION_PATH \
    --filtered_annotation_save_path $FILTERED_ANNOTATION_SAVE_PATH \
    --filter_num $FILTER_NUM \
    --gamma 1.0 \
    --k_nearest 10 \
    --use_fallback $USE_FALLBACK

echo "Data selection complete! Filtered dataset saved to $FILTERED_ANNOTATION_SAVE_PATH"