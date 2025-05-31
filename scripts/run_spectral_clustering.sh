#!/bin/bash

# Set feature type and clustering parameters
FEATURE_TYPE="clip+scores"  # Options: clip, scores, clip+scores
N_CLUSTERS=20
N_NEIGHBORS=30
GPU_IDS="0,1,2,3"  # GPU IDs to use, adjust based on available GPUs
CHUNK_SIZE=5000    # Data chunk size for each GPU

# Create results directory if it doesn't exist
mkdir -p ./data/results

# Execute multi-GPU spectral clustering
echo "Running multi-GPU spectral clustering..."
python ./data_process/spectral_clustering.py \
    --features_path ./data/scores/llava_clip_feature.pt \
    --n_clusters $N_CLUSTERS \
    --save_path ./data/results/spectral_clustering_${FEATURE_TYPE}_${N_CLUSTERS}_multi_gpu.json \
    --feature_type $FEATURE_TYPE \
    --n_neighbors $N_NEIGHBORS \
    --gpu_ids $GPU_IDS \
    --chunk_size $CHUNK_SIZE

echo "Spectral clustering completed! Results saved at: ./data/spectral_clustering_${FEATURE_TYPE}_${N_CLUSTERS}_multi_gpu.json"