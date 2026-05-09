"""
COIDO Stage 2: Data Selection.

Score the full dataset using the trained scorer,
then select per-dataset subsets based on CoIDO scores.
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
from coido_scorer_model import (
    LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback,
)
import tqdm
from transformers.modeling_utils import load_sharded_checkpoint
import torch.nn as nn


def load_stage1_model(
    model_path, feature_extractor_setting, device_map="auto", device="cuda", **kwargs
):
    """
    
    Args:
        model_path: 
        feature_extractor_setting: ('clip+scores')
        device_map: 
        device: 
        **kwargs: 
    Returns:
        
    """
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    kwargs["torch_dtype"] = torch.float16

    print(f": {model_path}")
    print(f": {feature_extractor_setting}")

    if feature_extractor_setting == "clip+scores":
        from coido_scorer_model import (
            LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback
        )
        print("Fallback...")
        model = LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        print("Fallback")
    else:
        print("Unknown feature extractor setting: ", feature_extractor_setting)
        raise NotImplementedError

    print(f": {type(model).__name__}")
    
    try:
        if hasattr(model, "hidden_size"):
            print(f"hidden_size: {model.hidden_size}")
        else:
            for name, module in model.named_modules():
                if isinstance(module, nn.LayerNorm):
                    print(f"LayerNorm {name} normalized_shape: {module.normalized_shape}")
                    break
    except Exception as e:
        print(f": {e}")
    
    return model


def load_scores(score_names):
    """
    
    
    Args:
        score_names: 
    Returns:
        
    """
    def norm_scores(score_dict: dict):
        """[-1, 1]"""
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


def get_difficulty_score(
    model_path: str, feature_extractor_setting: str, save_path: str
):
    """
    
    
    ,    
    Args:
        model_path: 
        feature_extractor_setting: 
        save_path: 
    Returns:
        
    """
    if os.path.exists(save_path):
        print("Difficulty already exists, generation skipped.")
        with open(save_path, "r") as f:
            difficulty_dict = json.load(f)
        return difficulty_dict

    print("Loading stage 1 model...", flush=True)
    model = load_stage1_model(model_path, feature_extractor_setting)
    print("Model loaded.", flush=True)

    if feature_extractor_setting == "clip+scores":
        return produce_clip_scores_difficulty(model, save_path)
    else:
        raise NotImplementedError(f"Unknown feature extractor setting: {feature_extractor_setting}")


def produce_clip_scores_difficulty(model, save_path: str, batch_size: int = 256):
    """
    CLIP
    
    Args:
        model: 
        save_path: 
        batch_size: ,128
    Returns:
        
    """
    difficulty_dict = {}
    
    model_type = type(model).__name__
    print(f"\n: {model_type}")
    is_fallback = "Fallback" in model_type
    if is_fallback:
        print("Fallback,...")
    
    print("CLIP...")
    clip_feat = torch.load("./data/scores/llava_clip_feature.pt")
    print(f"CLIP, {len(clip_feat)} ")
    
    score_names = [
        "./data/scores/llava_imagereward.json",
        "./data/scores/llava_clipscore.json",
        "./data/scores/deepseek-chat_single/processed_score.json",
    ]
    
    score_dicts = []
    print("...")
    for score_name in score_names:
        try:
            with open(score_name, "r") as f:
                score_dicts.append(json.load(f))
            print(f"- : {score_name}")
        except Exception as e:
            print(f" {score_name} : {e}")
            score_dicts.append({})

    if len(score_dicts) != 3:
        raise ValueError("3!")

    skipped_samples = []
    error_counts = {"missing_score": 0, "dimension_mismatch": 0, "other_errors": 0}
    
    try:
        dtype = model.get_score_net_dtype()
        print(f": {dtype}")
    except AttributeError:
        dtype = torch.float16
        print(f"get_score_net_dtype,: {dtype}")
    
    all_sample_ids = list(clip_feat.keys())
    total_samples = len(all_sample_ids)
    print(f",: {total_samples},: {batch_size}")
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm.tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_sample_ids = all_sample_ids[start_idx:end_idx]
        
        batch_clip_features = []
        batch_score_features = []
        valid_indices = []
        batch_original_ids = []
        
        for batch_pos, unique_idx_str in enumerate(batch_sample_ids):
            sample_id_str = str(unique_idx_str)
            batch_original_ids.append(sample_id_str)
            
            try:
                clip_features = clip_feat[sample_id_str].cuda().to(dtype=dtype)
                
                score_values = []
                
                if sample_id_str in score_dicts[0]:
                    val = score_dicts[0][sample_id_str]
                    score_values.append(float(val[0] if isinstance(val, list) else val))
                else:
                    score_values.append(0.5)
                    error_counts["missing_score"] += 1
                    skipped_samples.append(sample_id_str + "_img_reward")

                if sample_id_str in score_dicts[1]:
                     val = score_dicts[1][sample_id_str]
                     score_values.append(float(val[0] if isinstance(val, list) else val))
                else:
                     score_values.append(0.5)
                     error_counts["missing_score"] += 1
                     skipped_samples.append(sample_id_str + "_clip_score")

                if sample_id_str in score_dicts[2]:
                    val = score_dicts[2][sample_id_str]
                    if isinstance(val, list) and len(val) >= 3:
                        score_values.extend([float(v) for v in val[:3]])
                    else:
                        score_values.extend([5.0, 5.0, 5.0])
                        error_counts["dimension_mismatch"] += 1
                        skipped_samples.append(sample_id_str + "_deepseek_dim")
                else:
                     score_values.extend([5.0, 5.0, 5.0])
                     error_counts["missing_score"] += 1
                     skipped_samples.append(sample_id_str + "_deepseek")
                     
                if len(score_values) != 5:
                    error_counts["dimension_mismatch"] += 1
                    continue

                batch_clip_features.append(clip_features)
                batch_score_features.append(torch.tensor(score_values, dtype=dtype, device="cuda"))
                valid_indices.append(batch_pos)
                
            except Exception as e:
                print(f" {sample_id_str} : {str(e)}")
                error_counts["other_errors"] += 1
                continue
        
        if len(batch_clip_features) == 0:
            continue
            
        batch_clip_tensor = torch.stack(batch_clip_features, dim=0)
        batch_score_tensor = torch.stack(batch_score_features, dim=0)
        
        batch_combined_features = torch.cat([batch_clip_tensor, batch_score_tensor], dim=1)
        
        try:
            with torch.no_grad():
                batch_weights = model.predict_weights(batch_combined_features).squeeze()
                
                if len(batch_clip_features) == 1:
                    batch_weights = batch_weights.unsqueeze(0)
                
                for i, batch_pos in enumerate(valid_indices):
                    sample_id = batch_original_ids[batch_pos]
                    difficulty_dict[sample_id] = -batch_weights[i].item()
        except Exception as e:
            print(f": {str(e)}")
            print(f"  - : {batch_combined_features.shape}")
            error_counts["other_errors"] += len(valid_indices)
            
            print(",...")
            for i, batch_pos in enumerate(valid_indices):
                sample_id = batch_original_ids[batch_pos]
                single_feature = batch_combined_features[i:i+1]
                
                try:
                    with torch.no_grad():
                        weight = model.predict_weights(single_feature).item()
                    difficulty_dict[sample_id] = -weight
                except Exception as e2:
                    print(f" {sample_id} : {str(e2)}")
                    error_counts["other_errors"] += 1
                    continue

    print("\n,:")
    print(f"- : {error_counts['missing_score']} ")
    print(f"- : {error_counts['dimension_mismatch']} ")
    print(f"- : {error_counts['other_errors']} ")
    print(f": {len(difficulty_dict)} ")
    if skipped_samples:
        print(f"ID: {skipped_samples[:10]}...") 

    if len(difficulty_dict) == 0 and len(clip_feat) > 0:
         raise RuntimeError("!")

    print(f" {save_path}...")
    with open(save_path, "w") as f:
        json.dump(difficulty_dict, f)

    print(f"CLIP+Scores, {len(difficulty_dict)} ")
    return difficulty_dict


def dist_filter_with_dataset(
    raw_annotation_path, 
    difficulty_dict, 
    filter_num, 
    save_path,
):
    """
    ,    
    Args:
        raw_annotation_path: 
        difficulty_dict: 
        filter_num: 
        save_path: 
    """
    if os.path.exists(save_path):
        print(f": {save_path}")
        return

    print(f": {raw_annotation_path}")
    try:
        with open(raw_annotation_path, "r") as f:
            raw_annotation = json.load(f)
        print(f", {len(raw_annotation)} ")
    except Exception as e:
        print(f": {e}")
        return
    
    print("...")
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
            print(f" {idx} : {e}")
            continue
    
    if missing_dataset_key > 0:
        print(f":  {missing_dataset_key} 'dataset','unknown'")
    
    print(":")
    for dataset_name, samples in dataset_groups.items():
        print(f"  - {dataset_name}: {len(samples)} ")
    
    total_samples = len(raw_annotation)
    dataset_quotas = {
        dataset_name: max(1, int(filter_num * len(samples) / total_samples))
        for dataset_name, samples in dataset_groups.items()
    }
    
    total_quota = sum(dataset_quotas.values())
    if total_quota < filter_num:
        largest_dataset = max(dataset_groups.items(), key=lambda x: len(x[1]))[0]
        dataset_quotas[largest_dataset] += (filter_num - total_quota)
        print(f", '{largest_dataset}'  {filter_num - total_quota} ")
    elif total_quota > filter_num:
        largest_dataset = max(dataset_groups.items(), key=lambda x: len(x[1]))[0]
        dataset_quotas[largest_dataset] -= (total_quota - filter_num)
        print(f", '{largest_dataset}'  {total_quota - filter_num} ")
    
    print(":")
    for dataset_name, quota in dataset_quotas.items():
        print(f"  - {dataset_name}: {quota} ")
    
    print("...")
    new_annotation = []
    missing_difficulty_samples = 0
    
    for dataset_name, quota in dataset_quotas.items():
        print(f" '{dataset_name}'...")
        dataset_samples = dataset_groups[dataset_name]
        
        valid_samples = []
        for sample_id in dataset_samples:
            if sample_id in difficulty_dict:
                valid_samples.append(sample_id)
            else:
                missing_difficulty_samples += 1
        
        print(f"  -  '{dataset_name}'  {len(valid_samples)}/{len(dataset_samples)} ")
        
        if len(valid_samples) == 0:
            print(f"  - :  '{dataset_name}' ,")
            continue
        
        try:
            sorted_samples = sorted(
                [(sample_id, difficulty_dict[sample_id]) 
                for sample_id in valid_samples],
                key=lambda x: x[1],
                reverse=True
            )
        except Exception as e:
            print(f"  - : {e}")
            continue
        
        actual_quota = min(quota, len(sorted_samples))
        if actual_quota < quota:
            print(f"  - :  '{dataset_name}' , {actual_quota}/{quota} ")
            
        selected_samples = sorted_samples[:actual_quota]
        print(f"  -  '{dataset_name}'  {len(selected_samples)} ")
        
        for sample_id, _ in selected_samples:
            try:
                example = raw_annotation[int(sample_id)]
                new_annotation.append(example)
                
                difficulty_dict.pop(sample_id)
            except Exception as e:
                print(f"  -  {sample_id} : {e}")
                continue
    
    if missing_difficulty_samples > 0:
        print(f":  {missing_difficulty_samples} ")
    
    print(f" {len(new_annotation)} , {save_path}")
    try:
        with open(save_path, "w") as f:
            json.dump(new_annotation, f)
        print(f"")
    except Exception as e:
        print(f": {e}")
        return

    selected_counts = {}
    for item in new_annotation:
        dataset_name = item.get("dataset", "unknown")
        selected_counts[dataset_name] = selected_counts.get(dataset_name, 0) + 1
    
    print(":")
    for dataset_name, count in selected_counts.items():
        print(f"  - {dataset_name}: {count} ")
    
    print(f"! {len(new_annotation)}/{filter_num} ")


if __name__ == "__main__":
    """:"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_model_path", type=str)

    parser.add_argument("--result_dir", type=str, default="./data/results")
    parser.add_argument("--difficulty_save_name", type=str)

    parser.add_argument(
        "--raw_annotation_path",
        type=str,
        default="./data/llava_v1_5_665k_add_idx.json",
    )
    parser.add_argument("--filtered_annotation_save_path", type=str)
    parser.add_argument("--filter_num", type=int)
    
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    difficulty_dict = get_difficulty_score(
        args.stage1_model_path,
        "clip+scores",
        os.path.join(args.result_dir, args.difficulty_save_name),
    )
    
    dist_filter_with_dataset(
        args.raw_annotation_path,
        difficulty_dict,
        args.filter_num,
        args.filtered_annotation_save_path,
    )
