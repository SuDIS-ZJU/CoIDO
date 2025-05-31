#!/bin/bash
# CoIDO Stage 2 Data Filtering Script

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Set paths
STAGE1_MODEL_PATH="./data/checkpoints/coido_stage1_20CLS/checkpoint-2000/"
FEATURE_TYPE="clip+scores"
RAW_ANNOTATION_PATH="./data/training_data.json"
RESULT_DIR="./data/difficulty/"
DATA_DIR="./data/"
DIFFICULTY_SAVE_NAME="difficulty_${FEATURE_TYPE}_dataset_20CLS.json"
DIFFICULTY_SAVE_PATH="${RESULT_DIR}/${DIFFICULTY_SAVE_NAME}"
FILTERED_ANNOTATION_SAVE_PATH="${DATA_DIR}/filtered_training_data_20CLS.json"
FILTER_NUM=133040  # Number of samples to filter

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p $RESULT_DIR
mkdir -p $DATA_DIR

# Set directory permissions
echo "Setting directory permissions..."
chmod -R 755 $RESULT_DIR
chmod -R 755 $DATA_DIR

# Run stage2.py for data filtering (dataset-based approach)
echo "Starting dataset-based data filtering..."
python "./coido/stage2.py" \
    --stage1_model_path $STAGE1_MODEL_PATH \
    --feature_extractor_setting $FEATURE_TYPE \
    --result_dir $RESULT_DIR \
    --difficulty_save_name $DIFFICULTY_SAVE_NAME \
    --raw_annotation_path $RAW_ANNOTATION_PATH \
    --filtered_annotation_save_path $FILTERED_ANNOTATION_SAVE_PATH \
    --filter_num $FILTER_NUM

# Ensure output files have appropriate permissions
if [ -f "$FILTERED_ANNOTATION_SAVE_PATH" ]; then
    echo "Setting output file permissions..."
    chmod 644 $FILTERED_ANNOTATION_SAVE_PATH
    echo "File successfully saved and permissions set"
else
    echo "Warning: File was not successfully created"
fi

echo "Dataset-based data filtering completed! Filtered data saved to $FILTERED_ANNOTATION_SAVE_PATH" 