import argparse
import torch
import numpy as np
from sklearn.cluster import SpectralClustering
import json
import os
from tqdm import tqdm
import signal
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def signal_handler(signum, frame):
    """Handle Ctrl+C signal for graceful termination."""
    sys.exit(0)

def process_chunk(features, args, device_id):
    """Process a single data chunk for clustering."""
    # Move features to specified GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        features = torch.tensor(features, dtype=torch.float32).cuda(device_id)
    
    # Compute similarity matrix on GPU
    similarities = torch.nn.functional.cosine_similarity(
        features.unsqueeze(1), 
        features.unsqueeze(0), 
        dim=2
    )
    
    # Handle invalid values
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Ensure similarity matrix is symmetric
    similarities = (similarities + similarities.t()) / 2
    
    # Clamp values to [-1, 1] range
    similarities = torch.clamp(similarities, min=-1.0, max=1.0)
    
    # Convert similarity to distance (ensure non-negative)
    distances = (1 - similarities) / 2  # Map [-1,1] to [0,1]
    distances = torch.clamp(distances, min=0.0, max=1.0)
    
    # Build KNN graph
    k = min(args.n_neighbors, distances.shape[0] - 1)
    _, indices = torch.topk(distances, k=k, dim=1, largest=False)
    
    # Build sparse similarity matrix
    sparse_similarities = torch.zeros_like(distances)
    for i in range(distances.shape[0]):
        sparse_similarities[i, indices[i]] = 1 - distances[i, indices[i]]
    
    # Ensure symmetry
    sparse_similarities = (sparse_similarities + sparse_similarities.t()) / 2
    
    # Move similarity matrix back to CPU for spectral clustering
    sparse_similarities = sparse_similarities.cpu().numpy()
    
    # Perform spectral clustering
    spectral = SpectralClustering(
        n_clusters=args.n_clusters,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1,
        assign_labels='discretize'
    )
    
    try:
        labels = spectral.fit_predict(sparse_similarities)
        return labels, "spectral"
    except Exception:
        # If spectral clustering fails, fallback to K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=args.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(features.cpu().numpy())
        return labels, "kmeans"

def main():
    """Main function for performing spectral clustering."""
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Multi-GPU spectral clustering")
    parser.add_argument(
        "--features_path", 
        type=str, 
        required=True,
        help="Path to CLIP features or combined features"
    )
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=10,
        help="Number of clusters"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Path to save clustering results"
    )
    parser.add_argument(
        "--feature_type", 
        type=str, 
        choices=["clip", "scores", "clip+scores"],
        default="clip", 
        help="Type of features to use"
    )
    parser.add_argument(
        "--n_neighbors", 
        type=int, 
        default=30,
        help="Number of neighbors for similarity graph construction"
    )
    parser.add_argument(
        "--gpu_ids", 
        type=str, 
        default="0",
        help="GPU IDs to use, comma-separated (e.g., '0,1,2')"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=5000,
        help="Data chunk size for each GPU"
    )
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    try:
        # Load different features based on feature type
        if args.feature_type == "clip":
            # Load CLIP features
            features_dict = torch.load(args.features_path)
            image_ids = list(features_dict.keys())
            features = []
            for id in tqdm(image_ids, desc="Loading CLIP features"):
                features.append(features_dict[id])
            features = torch.stack(features).cpu().numpy()
        
        elif args.feature_type == "scores":
            # Load score features
            score_names = [
                "./data/scores/llava_clipscore.json",
                "./data/scores/llava_imagereward.json",
                "./data/scores/processed_score.json",
            ]
            
            # Load and normalize scores
            score_dicts = []
            for score_name in tqdm(score_names, desc="Loading score features"):
                with open(score_name, "r") as f:
                    score_dict = json.load(f)
                    min_score = min(score_dict.values())
                    max_score = max(score_dict.values())
                    normed_score_dict = {
                        unique_idx: (score - min_score) / (max_score - min_score) * 2 - 1
                        for unique_idx, score in score_dict.items()
                    }
                    score_dicts.append(normed_score_dict)
            
            # Get all sample IDs
            image_ids = list(score_dicts[0].keys())
            
            # Build feature matrix
            features = []
            for id in tqdm(image_ids, desc="Building score feature matrix"):
                feat = [score_dict[id] for score_dict in score_dicts]
                features.append(feat)
            features = np.array(features)
        
        elif args.feature_type == "clip+scores":
            # Load CLIP features
            clip_features_dict = torch.load("./data/scores/llava_clip_feature.pt")
            
            # Load score features
            score_names = [
                "./data/scores/llava_clipscore.json",
                "./data/scores/llava_imagereward.json",
                "./data/scores/processed_score.json",
            ]
            
            # Load and normalize scores
            score_dicts = []
            for score_name in tqdm(score_names, desc="Loading score features"):
                with open(score_name, "r") as f:
                    score_dict = json.load(f)
                    min_score = min(score_dict.values())
                    max_score = max(score_dict.values())
                    normed_score_dict = {
                        unique_idx: (score - min_score) / (max_score - min_score) * 2 - 1
                        for unique_idx, score in score_dict.items()
                    }
                    score_dicts.append(normed_score_dict)
            
            # Get common sample IDs
            common_ids = set(clip_features_dict.keys())
            for score_dict in score_dicts:
                common_ids = common_ids.intersection(set(score_dict.keys()))
            image_ids = sorted(list(common_ids))
            
            # Build combined feature matrix
            features = []
            for id in tqdm(image_ids, desc="Building combined feature matrix"):
                clip_feat = clip_features_dict[id].cpu().numpy()
                score_feat = np.array([score_dict[id] for score_dict in score_dicts])
                
                # Normalize CLIP features
                clip_feat = clip_feat / np.linalg.norm(clip_feat)
                
                # Normalize score features
                score_feat = (score_feat - score_feat.mean()) / score_feat.std()
                
                # Combine features with weights
                clip_weight = 0.7
                score_weight = 0.3
                combined_feat = np.concatenate([
                    clip_feat * clip_weight,
                    score_feat * score_weight
                ])
                features.append(combined_feat)
            
            features = np.array(features)
        
        # Split features into chunks
        n_samples = len(features)
        chunks = []
        for i in range(0, n_samples, args.chunk_size):
            end_idx = min(i + args.chunk_size, n_samples)
            chunks.append((features[i:end_idx], image_ids[i:end_idx]))
        
        # Process each chunk and collect results
        all_labels = []
        all_processed_ids = []
        clustering_methods = []
        
        for chunk_idx, (chunk_features, chunk_ids) in enumerate(chunks):
            # Select GPU
            device_id = gpu_ids[chunk_idx % len(gpu_ids)]
            
            # Check for invalid values in data chunk
            if np.any(np.isnan(chunk_features)) or np.any(np.isinf(chunk_features)):
                chunk_features = np.nan_to_num(chunk_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Process current chunk
            chunk_labels, method = process_chunk(chunk_features, args, device_id)
            
            all_labels.extend(chunk_labels)
            all_processed_ids.extend(chunk_ids)
            clustering_methods.append(method)
        
        # Organize final clustering results
        clustered_images = {}
        for i, label in enumerate(all_labels):
            label_str = str(int(label))
            if label_str not in clustered_images:
                clustered_images[label_str] = []
            clustered_images[label_str].append(all_processed_ids[i])
        
        # Save clustering results
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'w') as f:
            json.dump(clustered_images, f, indent=4)
        
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    main()