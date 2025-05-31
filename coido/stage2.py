"""
CoIDO Stage 2 Processing Script

Main functionalities:
1. Load the trained model from stage 1
2. Calculate difficulty scores for samples
3. Filter data based on difficulty scores and feature similarity
4. Generate filtered dataset
"""

import sys

sys.path.append("./LLaVA")

from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms
import json
import argparse
import tqdm
from coido_model import (
    LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback,
)
import tqdm
from transformers.modeling_utils import load_sharded_checkpoint
import torch.nn as nn
from peft import PeftModel, PeftConfig


def load_stage1_model(
    model_path, feature_extractor_setting, device_map="auto", device="cuda", use_fallback=False, **kwargs
):
    """
    Load the trained model from stage 1 (adapted for LoRA loading)
    
    Args:
        model_path: LoRA adapter path
        feature_extractor_setting: Feature extractor setting ('clip', 'scores', or 'clip+scores')
        device_map: Device mapping method
        device: Device to use
        use_fallback: Whether to use fallback model
        **kwargs: Other parameters
    Returns:
        Loaded model instance
    """
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    # LoRA models usually use the same torch_dtype as the base model, or specify it during loading
    # If the base model is float16, adapters usually are too, keeping float16 setting here
    kwargs["torch_dtype"] = torch.float16 

    print(f"Loading LoRA model from adapter path: {model_path}")
    print(f"Feature extractor setting: {feature_extractor_setting}")
    print(f"Using Fallback model (based on parameter): {use_fallback}")

    # 1. Load PeftConfig from adapter path to get base model information
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_name_or_path = peft_config.base_model_name_or_path
        print(f"Got base model path from adapter_config.json: {base_model_name_or_path}")
    except Exception as e:
        print(f"Error: Unable to load PeftConfig (adapter_config.json) from {model_path}. "
              f"Please ensure it's a valid LoRA adapter path and contains adapter_config.json. Error: {e}")
        raise

    # Load corresponding model based on feature extractor setting
    # Since we simplified to only use clip+scores, we only need ClipScoresFallback
    base_model_class = LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback
    model_description = "LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback"

    print(f"Loading base model: {base_model_name_or_path} using {model_description}")
    base_model_instance = base_model_class.from_pretrained(
        base_model_name_or_path, low_cpu_mem_usage=True, **kwargs
    )
    
    print(f"Loading LoRA adapter: {model_path} to {model_description}")
    # For inference, usually set is_trainable=False
    model = PeftModel.from_pretrained(base_model_instance, model_path, is_trainable=False, device_map=device_map if device_map else "auto")

    print(f"Successfully loaded LoRA model: {type(model).__name__} (base model: {type(base_model_instance).__name__})")
    
    # When accessing original model methods and attributes, need to go through model.base_model
    # For example: model.base_model.get_score_net_dtype()
    # predict_weights should also be called on model.base_model, or PeftModel needs to properly proxy it.
    # If predict_weights is a custom method, ensure it remains accessible or is correctly called after PeftModel wrapping.

    # Try to detect hidden_size parameter in model (now need to access through base_model)
    try:
        # PeftModel wraps base_model_instance
        effective_model_for_attrs = model.base_model 
        if hasattr(effective_model_for_attrs, "hidden_size"):
             print(f"Base model hidden_size: {effective_model_for_attrs.hidden_size}")
        # LLaVA models usually have a .model attribute
        elif hasattr(effective_model_for_attrs, "model") and hasattr(effective_model_for_attrs.model, "hidden_size"):
             print(f"Base model (model.base_model.model) hidden_size: {effective_model_for_attrs.model.hidden_size}")
        else:
            for name, module in effective_model_for_attrs.named_modules():
                if isinstance(module, nn.LayerNorm):
                    print(f"Detected base model LayerNorm layer {name} with normalized_shape: {module.normalized_shape}")
                    break
    except Exception as e:
        print(f"Error checking base model parameter dimensions: {e}")
    
    return model


def load_scores(score_names):
    """
    Load and normalize scores
    
    Args:
        score_names: List of score file paths
    Returns:
        List of normalized score dictionaries
    """
    def norm_scores(score_dict: dict):
        """Normalize scores to [-1, 1] range"""
        min_score = min(score_dict.values())
        max_score = max(score_dict.values())
        normed_score_dict = {
            i[0]: (i[1] - min_score) / (max_score - min_score) * 2 - 1
            for i in score_dict.items()
        }
        return normed_score_dict

    score_dicts = []

    for score_name in score_names:
        with open(score_name, "r") as f:
            score_dict = json.load(f)
            score_dicts.append(norm_scores(score_dict))

    return score_dicts


def produce_scores_difficulty(model, save_path: str):
    """
    Generate sample difficulty scores based on precomputed scores
    
    Args:
        model: Model instance
        save_path: Path to save difficulty scores
    Returns:
        Difficulty score dictionary
    """
    difficulty_dict = {}

    # Load precomputed score files
    score_dicts = [
        "./data/scores/llava_imagereward.json",
        "./data/scores/llava_imagereward.json", 
        "./data/scores/deepseek-chat/processed_score.json",
    ]

    # Calculate difficulty score for each sample
    for unique_idx in score_dicts[0]:
        scores = [[score_dict[str(unique_idx)] for score_dict in score_dicts]]
        scores = torch.tensor(scores).cuda().half()
        # Use negative value of model-predicted weights as difficulty score
        difficulty_dict[unique_idx] = -model.predict_weights(scores).item()

    # Save difficulty scores
    with open(save_path, "w") as f:
        json.dump(difficulty_dict, f)

    print("Scores difficulty generated and saved.")
    return difficulty_dict


def produce_clip_difficulty(model, save_path: str):
    """
    Generate sample difficulty scores based on CLIP features
    
    Args:
        model: Model instance
        save_path: Path to save difficulty scores
    Returns:
        Difficulty score dictionary
    """
    clip_feat = torch.load("./data/scores/llava_clip_feature.pt")
    print("CLIP features loaded")

    difficulty_dict = {}
    underlying_model = model.base_model if hasattr(model, 'base_model') else model
    dtype = underlying_model.get_score_net_dtype()

    for unique_idx in tqdm.tqdm(clip_feat):
        scores = clip_feat[unique_idx].cuda().to(dtype=dtype).unsqueeze(0)
        difficulty_dict[unique_idx] = -underlying_model.predict_weights(scores).item()

    with open(save_path, "w") as f:
        json.dump(difficulty_dict, f)

    print("CLIP difficulty generated and saved")
    return difficulty_dict


def get_difficulty_score(
    model_path: str, feature_extractor_setting: str, save_path: str, use_fallback: bool = False
):
    """
    Get difficulty scores for samples
    
    If difficulty score file already exists, load it directly; otherwise compute anew
    
    Args:
        model_path: Model path
        feature_extractor_setting: Feature extractor setting
        save_path: Path to save difficulty scores
        use_fallback: Whether to use fallback model
    Returns:
        Difficulty score dictionary
    """
    # Load directly if difficulty score file already exists
    if os.path.exists(save_path):
        print("Difficulty already exists, generation skipped.")
        with open(save_path, "r") as f:
            difficulty_dict = json.load(f)
        return difficulty_dict

    # Load model and generate difficulty scores
    print("Loading stage 1 model...", flush=True)
    model = load_stage1_model(model_path, feature_extractor_setting, use_fallback=use_fallback)
    print("Model loaded.", flush=True)

    if feature_extractor_setting == "scores":
        return produce_scores_difficulty(model, save_path)
    elif feature_extractor_setting == "clip":
        return produce_clip_difficulty(model, save_path)
    elif feature_extractor_setting == "clip+scores":
        return produce_clip_scores_difficulty(model, save_path)
    else:
        raise NotImplementedError(f"Unknown feature extractor setting: {feature_extractor_setting}")


def _normalize_score_dict(score_dict: dict, norm_range: tuple = (-1, 1)) -> dict:
    """Normalize scores in score dictionary to specified range"""
    if not score_dict:
        return {}
    min_val = min(score_dict.values())
    max_val = max(score_dict.values())
    
    # Prevent division by zero
    if max_val == min_val:
        # If all values are the same, set them based on the middle of the range or 0
        # For example, for [-1, 1], the middle value is 0. For [0,1], the middle value is 0.5
        # Here we assume if they're all the same and the range includes 0, set them to 0, otherwise set to range lower bound
        # Or more simply, map them all to the range lower bound or middle value
        # If min_val == max_val, all values are the same.
        # If range is [-1, 1], normalized value should be 0.0 (midpoint)
        # If range is [0, 1], normalized value should be 0.5 (midpoint)
        # For generality and to avoid assumptions, if all values are the same, we map them to (norm_range[0] + norm_range[1]) / 2
        # Or, if they're already the expected value (e.g., all 0), keep them unchanged.
        # A simple approach is, if all are the same, return norm_range[0]
        # Or if all are the same, return (norm_range[0] + norm_range[1]) / 2
        # Here we adopt mapping to the range midpoint
        target_val = (norm_range[0] + norm_range[1]) / 2.0
        return {k: target_val for k in score_dict}

    norm_min, norm_max = norm_range
    return {
        k: norm_min + (v - min_val) * (norm_max - norm_min) / (max_val - min_val)
        for k, v in score_dict.items()
    }


def produce_clip_scores_difficulty(model, save_path: str, batch_size: int = 256):
    """
    Generate sample difficulty scores based on CLIP features and precomputed scores (adapted to score processing logic in trainer.py)
    
    Args:
        model: Model instance
        save_path: Path to save difficulty scores
        batch_size: Batch size
    Returns:
        Difficulty score dictionary
    """
    difficulty_dict = {}
    
    model_type = type(model).__name__
    print(f"\nUsing model type: {model_type}")
    
    print("Loading CLIP features...")
    clip_feat = torch.load("./data/scores/llava_clip_feature.pt")
    print(f"Successfully loaded CLIP features for {len(clip_feat)} samples")
    
    # Change score file paths and order to match trainer.py
    # Order: ClipScore, ImageReward, DeepSeek_single
    score_files_config = [
        {"path": "./data/scores/llava_clipscore.json", "name": "clipscore", "dim": 1, "default": 0.0}, # Assume normalized missing default value is 0
        {"path": "./data/scores/llava_imagereward.json", "name": "imagereward", "dim": 1, "default": 0.0},
        {"path": "./data/scores/deepseek-chat/processed_score.json", "name": "deepseek_single", "dim": 1, "default": 0.0}
    ]
    
    loaded_scores = []
    print("Loading and normalizing raw score files (range [-1, 1])...")
    for config in score_files_config:
        try:
            with open(config["path"], "r") as f:
                raw_score_dict = json.load(f)
            # Normalization processing, assume target range is [-1, 1]
            normalized_score_dict = _normalize_score_dict(raw_score_dict, norm_range=(-1, 1))
            loaded_scores.append(normalized_score_dict)
            print(f"- Loaded and normalized: {config['path']}")
        except Exception as e:
            print(f"Failed to load or normalize score file {config['path']}: {e}. Will use empty dictionary.")
            loaded_scores.append({}) # If loading fails, add empty dictionary

    if len(loaded_scores) != len(score_files_config):
        raise ValueError("Failed to successfully load all necessary score files!")

    skipped_samples_log = {}
    error_counts = {"missing_score_total": 0, "other_errors": 0}
    
    try:
        # For PeftModel, actual model is in .base_model
        dtype = model.base_model.get_score_net_dtype() if hasattr(model, 'base_model') else model.get_score_net_dtype()
        print(f"Using model-provided data type: {dtype}")
    except AttributeError:
        dtype = torch.float16
        print(f"Model does not provide get_score_net_dtype method, using default data type: {dtype}")
    
    all_sample_ids = list(clip_feat.keys()) # Assume all sample IDs are in clip_feat
    total_samples = len(all_sample_ids)
    print(f"Starting to process sample difficulty scores, total samples: {total_samples}, batch size: {batch_size}")
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm.tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_sample_ids = all_sample_ids[start_idx:end_idx]
        
        batch_clip_features_list = []
        batch_score_features_list = []
        valid_indices_in_batch = []
        batch_original_ids_list = []
        
        for batch_pos, unique_idx_str in enumerate(batch_sample_ids):
            sample_id_str = str(unique_idx_str) # Ensure it's a string
            batch_original_ids_list.append(sample_id_str)
            
            try:
                clip_features_tensor = clip_feat[sample_id_str].cuda().to(dtype=dtype)
                
                # Build 3-dimensional score features [ClipScore, ImageReward, DeepSeek_single]
                # All scores are normalized to [-1, 1]
                current_sample_scores = []
                missing_score_for_sample = False
                for i, score_dict in enumerate(loaded_scores):
                    score_conf = score_files_config[i]
                    if sample_id_str in score_dict:
                        # Ensure extracted value is a single number
                        val = score_dict[sample_id_str]
                        current_sample_scores.append(float(val[0] if isinstance(val, list) else val))
                    else:
                        current_sample_scores.append(float(score_conf["default"])) # Use configured default value
                        error_key = f"missing_{score_conf['name']}"
                        skipped_samples_log[error_key] = skipped_samples_log.get(error_key, 0) + 1
                        missing_score_for_sample = True
                
                if missing_score_for_sample:
                    error_counts["missing_score_total"] +=1

                if len(current_sample_scores) != 3:
                    # This should not happen if logic is correct
                    print(f"Warning: Sample {sample_id_str} score dimension is not 3, actual is {len(current_sample_scores)}. Skipping this sample.")
                    error_counts["other_errors"] += 1
                    continue

                batch_clip_features_list.append(clip_features_tensor)
                batch_score_features_list.append(torch.tensor(current_sample_scores, dtype=dtype, device="cuda"))
                valid_indices_in_batch.append(batch_pos)
                
            except Exception as e:
                print(f"Error processing sample {sample_id_str}: {str(e)}")
                error_counts["other_errors"] += 1
                continue
        
        if not batch_clip_features_list:
            continue
            
        batch_clip_tensor = torch.stack(batch_clip_features_list, dim=0)
        batch_score_tensor = torch.stack(batch_score_features_list, dim=0)
        
        # Concatenate CLIP (1536) and Scores (3) features
        # Fallback model internally separates and processes clip (1536) and scores (3)
        batch_combined_features = torch.cat([batch_clip_tensor, batch_score_tensor], dim=1) 
        
        try:
            with torch.no_grad():
                # For PeftModel, actual model is in .base_model
                predict_method = model.base_model.predict_weights if hasattr(model, 'base_model') else model.predict_weights
                batch_weights = predict_method(batch_combined_features).squeeze()
                
                if len(batch_clip_features_list) == 1:
                    batch_weights = batch_weights.unsqueeze(0) if batch_weights.ndim == 0 else batch_weights
                
                for i, original_batch_pos in enumerate(valid_indices_in_batch):
                    sample_id_for_dict = batch_original_ids_list[original_batch_pos]
                    difficulty_dict[sample_id_for_dict] = -batch_weights[i].item()
        except Exception as e:
            print(f"Error predicting weights: {str(e)}")
            print(f"  - Input feature shape: {batch_combined_features.shape}")
            error_counts["other_errors"] += len(valid_indices_in_batch)
            
            print("Attempting to process failed batch sample by sample...")
            for i, original_batch_pos in enumerate(valid_indices_in_batch):
                sample_id_for_dict = batch_original_ids_list[original_batch_pos]
                single_feature = batch_combined_features[i:i+1]
                try:
                    with torch.no_grad():
                        predict_method = model.base_model.predict_weights if hasattr(model, 'base_model') else model.predict_weights
                        weight = predict_method(single_feature).item()
                    difficulty_dict[sample_id_for_dict] = -weight
                except Exception as e2:
                    print(f"Error processing single sample {sample_id_for_dict}: {str(e2)}")
                    error_counts["other_errors"] += 1
                    continue

    print("\nProcessing completed, error statistics:")
    print(f"- Total times at least one score is missing: {error_counts['missing_score_total']}")
    for score_name_key, count in skipped_samples_log.items():
        print(f"  - {score_name_key} missing times: {count}")
    print(f"- Other processing errors: {error_counts['other_errors']} times")
    print(f"Successfully calculated difficulty scores for {len(difficulty_dict)} samples")

    if not difficulty_dict and total_samples > 0:
         print("Warning: All samples' difficulty scores calculation failed! Please check score files and model.")
         # raise RuntimeError("All samples' difficulty scores calculation failed!") # Commented out temporarily to allow saving empty dictionary even if failed

    print(f"Saving difficulty scores to {save_path}...")
    with open(save_path, "w") as f:
        json.dump(difficulty_dict, f)

    print(f"CLIP+Scores difficulty scores generated and saved, processed {len(difficulty_dict)} samples")
    return difficulty_dict


def dist_filter(
    raw_annotation_path, difficulty_dict, filter_num, save_path, gamma=1, k_nearest=10
):
    """
    Filter data based on difficulty scores and feature similarity
    
    Args:
        raw_annotation_path: Path to raw annotation file
        difficulty_dict: Difficulty score dictionary
        filter_num: Number of samples to filter out
        save_path: Path to save filtered annotation
        gamma: Similarity penalty coefficient
        k_nearest: Number of nearest neighbors to update
    """
    # If filtered annotation already exists, return directly
    if os.path.exists(save_path):
        print("Filtered annotation already exists.")
        return

    # Load raw annotation data
    with open(raw_annotation_path, "r") as f:
        raw_annotation = json.load(f)
    new_annotation = []

    # Load CLIP features and calculate feature matrix
    feat_dict = torch.load("./data/scores/llava_clip_feature.pt")
    feat_len = len(feat_dict)
    feat_matrix = torch.stack(
        [feat_dict[str(i)].cuda() for i in range(feat_len)], dim=0
    )
    # Calculate feature vector norm, used for cosine similarity calculation
    feat_matrix_norm = torch.norm(feat_matrix, dim=-1, keepdim=False)

    # Iterate to select samples
    for i in tqdm.tqdm(range(filter_num)):
        # Select sample with highest difficulty
        lst = sorted(difficulty_dict.items(), key=lambda x: x[1], reverse=True)
        unique_idx, difficulty = lst[0]

        # Add selected sample to new dataset
        example = raw_annotation[int(unique_idx)]
        example.pop("unique_idx")
        new_annotation.append(example)
        difficulty_dict.pop(unique_idx)

        # Calculate feature similarity with selected sample
        tgt_feat = feat_matrix[int(unique_idx)].unsqueeze(dim=0)
        tgt_norm = feat_matrix_norm[int(unique_idx)].unsqueeze(dim=0)
        # Calculate cosine similarity
        sims = (feat_matrix * tgt_feat).sum(dim=-1) / feat_matrix_norm / tgt_norm

        # Get k_nearest most similar samples
        sorted_sim, indices = torch.sort(sims, descending=True)
        success_cnt = 0

        # Update difficulty scores of similar samples
        for j in range(len(difficulty_dict)):
            if success_cnt >= k_nearest:
                break

            cur_unique_idx = str(indices[j].item())
            if cur_unique_idx not in difficulty_dict:
                continue

            # Update difficulty score based on similarity and difficulty
            cur_sim = sorted_sim[j].item()
            penalty = difficulty * (cur_sim**2) * gamma
            difficulty_dict[cur_unique_idx] -= penalty
            success_cnt += 1

        assert success_cnt == k_nearest

    # Save filtered dataset
    with open(save_path, "w") as f:
        json.dump(new_annotation, f)

    print("Annotation filtered and saved.")


def dist_filter_with_dataset(
    raw_annotation_path, 
    difficulty_dict, 
    filter_num, 
    save_path
):
    """
    Filter data based on difficulty scores, ensuring diversity across different datasets
    
    This function uses dataset-based filtering, which directly allocates quotas based on
    dataset proportions and selects the most difficult samples within each dataset.
    Unlike similarity-based filtering, this approach does not use gamma/k_nearest parameters.
    
    Args:
        raw_annotation_path: Path to raw annotation file
        difficulty_dict: Difficulty score dictionary
        filter_num: Number of samples to filter out
        save_path: Path to save filtered annotation
    """
    # If filtered annotation already exists, return directly
    if os.path.exists(save_path):
        print(f"Filtered annotation file already exists: {save_path}")
        return

    # Load raw annotation data
    print(f"Loading raw annotation data: {raw_annotation_path}")
    try:
        with open(raw_annotation_path, "r") as f:
            raw_annotation = json.load(f)
        print(f"Successfully loaded raw annotation data, total {len(raw_annotation)} samples")
    except Exception as e:
        print(f"Failed to load raw annotation data: {e}")
        return
    
    # Group by dataset and count samples in each dataset
    print("Grouping by dataset...")
    dataset_groups = {}
    missing_dataset_key = 0
    
    for idx, item in enumerate(raw_annotation):
        try:
            dataset_name = item.get("dataset", "unknown")
            if dataset_name == "unknown":
                missing_dataset_key += 1
                
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(str(idx))
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if missing_dataset_key > 0:
        print(f"Warning: {missing_dataset_key} samples are missing 'dataset' key, categorized as 'unknown'")
    
    # Print each dataset's sample count
    print("Each dataset sample count:")
    for dataset_name, samples in dataset_groups.items():
        print(f"  - {dataset_name}: {len(samples)} samples")
    
    # Calculate each dataset's sample selection quota
    total_samples = len(raw_annotation)
    dataset_quotas = {
        dataset_name: max(1, int(filter_num * len(samples) / total_samples))
        for dataset_name, samples in dataset_groups.items()
    }
    
    # Ensure quota total equals filter_num
    total_quota = sum(dataset_quotas.values())
    if total_quota < filter_num:
        # Add remaining quota to largest dataset
        largest_dataset = max(dataset_groups.items(), key=lambda x: len(x[1]))[0]
        dataset_quotas[largest_dataset] += (filter_num - total_quota)
        print(f"Quota total less than target number, added {filter_num - total_quota} quota to largest dataset '{largest_dataset}'")
    elif total_quota > filter_num:
        # Reduce extra quota from largest dataset
        largest_dataset = max(dataset_groups.items(), key=lambda x: len(x[1]))[0]
        dataset_quotas[largest_dataset] -= (total_quota - filter_num)
        print(f"Quota total greater than target number, reduced {total_quota - filter_num} quota from largest dataset '{largest_dataset}'")
    
    print("Each dataset quota:")
    for dataset_name, quota in dataset_quotas.items():
        print(f"  - {dataset_name}: {quota} samples")
    
    # Select samples with highest difficulty from each dataset
    print("Starting to select samples from each dataset...")
    new_annotation = []
    missing_difficulty_samples = 0
    
    for dataset_name, quota in dataset_quotas.items():
        print(f"Processing dataset '{dataset_name}'...")
        # Get samples in this dataset
        dataset_samples = dataset_groups[dataset_name]
        
        # Filter out samples not in difficulty_dict
        valid_samples = []
        for sample_id in dataset_samples:
            if sample_id in difficulty_dict:
                valid_samples.append(sample_id)
            else:
                missing_difficulty_samples += 1
        
        print(f"  - Dataset '{dataset_name}' has {len(valid_samples)}/{len(dataset_samples)} samples with difficulty scores")
        
        if len(valid_samples) == 0:
            print(f"  - Warning: Dataset '{dataset_name}' has no valid samples, skipping")
            continue
        
        # Sort by difficulty
        try:
            sorted_samples = sorted(
                [(sample_id, difficulty_dict[sample_id]) 
                for sample_id in valid_samples],
                key=lambda x: x[1],
                reverse=True  # Difficulty from high to low
            )
        except Exception as e:
            print(f"  - Error sorting samples: {e}")
            continue
        
        # Select top quota samples
        actual_quota = min(quota, len(sorted_samples))
        if actual_quota < quota:
            print(f"  - Warning: Insufficient valid samples in dataset '{dataset_name}', only selecting {actual_quota}/{quota} samples")
            
        selected_samples = sorted_samples[:actual_quota]
        print(f"  - Selected {len(selected_samples)} samples from dataset '{dataset_name}'")
        
        # Add to result
        for sample_id, _ in selected_samples:
            try:
                example = raw_annotation[int(sample_id)]
                new_annotation.append(example)
                
                # Remove selected sample from difficulty_dict
                difficulty_dict.pop(sample_id)
            except Exception as e:
                print(f"  - Error adding sample {sample_id}: {e}")
                continue
    
    if missing_difficulty_samples > 0:
        print(f"Warning: {missing_difficulty_samples} samples are missing difficulty scores")
    
    # Save filtered dataset
    print(f"Selected {len(new_annotation)} samples, saving to {save_path}")
    try:
        with open(save_path, "w") as f:
            json.dump(new_annotation, f)
        print(f"Successfully saved filtered dataset")
    except Exception as e:
        print(f"Failed to save filtered dataset: {e}")
        return

    # Print each dataset's actual selected sample count
    selected_counts = {}
    for item in new_annotation:
        dataset_name = item.get("dataset", "unknown")
        selected_counts[dataset_name] = selected_counts.get(dataset_name, 0) + 1
    
    print("Each dataset's actual selected sample count:")
    for dataset_name, count in selected_counts.items():
        print(f"  - {dataset_name}: {count} samples")
    
    print(f"Data filtering completed! Selected {len(new_annotation)}/{filter_num} samples")


if __name__ == "__main__":
    """Main function: Parse command line arguments and execute data filtering process"""
    parser = argparse.ArgumentParser()
    # Model related parameters
    parser.add_argument("--stage1_model_path", type=str)
    parser.add_argument(
        "--feature_extractor_setting", type=str, choices=["scores", "clip", "clip+scores"]
    )

    # Output related parameters
    parser.add_argument("--result_dir", type=str, default="./data/results")
    parser.add_argument("--difficulty_save_name", type=str)

    # Filter related parameters
    parser.add_argument(
        "--raw_annotation_path",
        type=str,
        default="./data/training_data.json",
    )
    parser.add_argument("--filtered_annotation_save_path", type=str)
    parser.add_argument("--filter_num", type=int)
    
    # Clustering related parameters (optional for future use)
    parser.add_argument("--use_clustering", action="store_true", help="Whether to use clustering for diversity selection")
    parser.add_argument("--clustering_results_path", type=str, help="Clustering results file path")
    
    args = parser.parse_args()

    # Create result directory
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # Always use fallback model since it's required for our setup
    use_fallback_bool = True
    print(f"Using Fallback model: {use_fallback_bool}")

    # Get difficulty scores
    difficulty_dict = get_difficulty_score(
        args.stage1_model_path,
        args.feature_extractor_setting,
        os.path.join(args.result_dir, args.difficulty_save_name),
        use_fallback=use_fallback_bool
    )
    
    # Use dataset-based filtering function (does not use gamma/k_nearest)
    dist_filter_with_dataset(
        args.raw_annotation_path,
        difficulty_dict,
        args.filter_num,
        args.filtered_annotation_save_path,
    )
