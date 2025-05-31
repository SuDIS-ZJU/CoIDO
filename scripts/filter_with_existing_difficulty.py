#!/usr/bin/env python
"""
使用已有难度分数文件按照数据集分组筛选样本

此脚本直接使用预先计算的难度分数文件，按照数据集分组筛选出指定数量的样本。
无需重新计算难度分数，直接从现有文件加载并进行筛选。
"""

import os
import json
import argparse
import tqdm
from collections import defaultdict


def dist_filter_with_dataset(
    raw_annotation_path, 
    difficulty_dict_path, 
    filter_num, 
    save_path
):
    """
    基于数据集来源和难度分数进行数据过滤，保证不同数据集的多样性
    
    Args:
        raw_annotation_path: 原始标注文件路径
        difficulty_dict_path: 已有难度分数字典的路径
        filter_num: 需要过滤得到的样本数量
        save_path: 过滤后数据的保存路径
    """
    # 如果已存在过滤后的文件则直接返回
    if os.path.exists(save_path):
        print(f"过滤后的标注文件已存在: {save_path}")
        return

    # 加载原始标注数据
    print(f"正在加载原始标注数据: {raw_annotation_path}")
    try:
        with open(raw_annotation_path, "r") as f:
            raw_annotation = json.load(f)
        print(f"成功加载原始标注数据，共 {len(raw_annotation)} 个样本")
    except Exception as e:
        print(f"加载原始标注数据失败: {e}")
        return
    
    # 加载已有的难度分数
    print(f"正在加载难度分数: {difficulty_dict_path}")
    try:
        with open(difficulty_dict_path, "r") as f:
            difficulty_dict = json.load(f)
        print(f"成功加载难度分数，共 {len(difficulty_dict)} 个样本")
    except Exception as e:
        print(f"加载难度分数失败: {e}")
        return
    
    # 按数据集分组并统计每个数据集的样本数
    print("正在按数据集分组...")
    dataset_groups = {}
    missing_dataset_key = 0
    
    for idx, item in enumerate(raw_annotation):
        try:
            # 确保样本ID在难度分数字典中
            sample_id = str(idx)
            if sample_id not in difficulty_dict:
                continue
                
            dataset_name = item.get("dataset", "unknown")
            if dataset_name == "unknown":
                missing_dataset_key += 1
                
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(sample_id)
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
            continue
    
    if missing_dataset_key > 0:
        print(f"警告: 有 {missing_dataset_key} 个样本缺少'dataset'键，已归类为'unknown'")
    
    # 打印各数据集样本数量
    print("各数据集样本数量:")
    for dataset_name, samples in dataset_groups.items():
        print(f"  - {dataset_name}: {len(samples)} 个样本")
    
    # 计算每个数据集应选择的样本数量
    total_samples = sum(len(samples) for samples in dataset_groups.values())
    dataset_quotas = {
        dataset_name: max(1, int(filter_num * len(samples) / total_samples))
        for dataset_name, samples in dataset_groups.items()
    }
    
    # 确保配额总和等于filter_num
    total_quota = sum(dataset_quotas.values())
    if total_quota < filter_num:
        # 为最大的数据集增加剩余配额
        largest_dataset = max(dataset_groups.items(), key=lambda x: len(x[1]))[0]
        dataset_quotas[largest_dataset] += (filter_num - total_quota)
        print(f"配额总和小于目标数量，已为最大数据集 '{largest_dataset}' 增加 {filter_num - total_quota} 个配额")
    elif total_quota > filter_num:
        # 从最大的数据集减少多余配额
        largest_dataset = max(dataset_groups.items(), key=lambda x: len(x[1]))[0]
        dataset_quotas[largest_dataset] -= (total_quota - filter_num)
        print(f"配额总和大于目标数量，已从最大数据集 '{largest_dataset}' 减少 {total_quota - filter_num} 个配额")
    
    print("各数据集配额:")
    for dataset_name, quota in dataset_quotas.items():
        print(f"  - {dataset_name}: {quota} 个样本")
    
    # 从每个数据集中选择难度最高的样本
    print("开始从各数据集中选择样本...")
    new_annotation = []
    missing_difficulty_samples = 0
    
    for dataset_name, quota in dataset_quotas.items():
        print(f"处理数据集 '{dataset_name}'...")
        # 获取该数据集中的样本
        dataset_samples = dataset_groups[dataset_name]
        
        # 过滤掉不在difficulty_dict中的样本
        valid_samples = []
        for sample_id in dataset_samples:
            if sample_id in difficulty_dict:
                valid_samples.append(sample_id)
            else:
                missing_difficulty_samples += 1
        
        print(f"  - 数据集 '{dataset_name}' 中有 {len(valid_samples)}/{len(dataset_samples)} 个样本有难度分数")
        
        if len(valid_samples) == 0:
            print(f"  - 警告: 数据集 '{dataset_name}' 中没有有效样本，跳过")
            continue
        
        # 按难度排序
        try:
            sorted_samples = sorted(
                [(sample_id, difficulty_dict[sample_id]) 
                for sample_id in valid_samples],
                key=lambda x: x[1],
                reverse=True  # 难度从高到低
            )
        except Exception as e:
            print(f"  - 排序样本时出错: {e}")
            continue
        
        # 选择前quota个样本
        actual_quota = min(quota, len(sorted_samples))
        if actual_quota < quota:
            print(f"  - 警告: 数据集 '{dataset_name}' 中有效样本数量不足，只能选择 {actual_quota}/{quota} 个样本")
            
        selected_samples = sorted_samples[:actual_quota]
        print(f"  - 已从数据集 '{dataset_name}' 中选择 {len(selected_samples)} 个样本")
        
        # 添加到结果中
        for sample_id, _ in selected_samples:
            try:
                example = raw_annotation[int(sample_id)]
                new_annotation.append(example)
            except Exception as e:
                print(f"  - 添加样本 {sample_id} 时出错: {e}")
                continue
    
    if missing_difficulty_samples > 0:
        print(f"警告: 有 {missing_difficulty_samples} 个样本没有难度分数")
    
    # 保存过滤后的数据集
    print(f"共选择了 {len(new_annotation)} 个样本，正在保存到 {save_path}")
    try:
        with open(save_path, "w") as f:
            json.dump(new_annotation, f)
        print(f"成功保存过滤后的数据集")
    except Exception as e:
        print(f"保存过滤后的数据集失败: {e}")
        return

    # 打印每个数据集实际选择的样本数量
    selected_counts = {}
    for item in new_annotation:
        dataset_name = item.get("dataset", "unknown")
        selected_counts[dataset_name] = selected_counts.get(dataset_name, 0) + 1
    
    print("各数据集实际选择的样本数量:")
    for dataset_name, count in sorted(selected_counts.items()):
        print(f"  - {dataset_name}: {count} 个样本")
    
    print(f"数据过滤完成！共选择了 {len(new_annotation)}/{filter_num} 个样本")


def main():
    parser = argparse.ArgumentParser(description="使用已有难度分数按数据集分组筛选样本")
    
    parser.add_argument("--difficulty_path", type=str, 
                        default="/data2/self-filter_ckpt/datasets/results/difficulty_clip+scores_dataset.json",
                        help="已有难度分数文件路径")
    
    parser.add_argument("--raw_annotation_path", type=str,
                        default="/home/yyc/Self-Filter/data/llava_v1_5_665k_add_idx_with_dataset.json",
                        help="原始标注文件路径")
    
    parser.add_argument("--save_path", type=str,
                        default="/data2/self-filter_ckpt/datasets/data/llava_v1_5_filtered_dataset_0.05select.json",
                        help="过滤后数据保存路径")
    
    parser.add_argument("--filter_num", type=int, default=33262,
                        help="需要筛选的样本数量")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 执行过滤
    dist_filter_with_dataset(
        args.raw_annotation_path,
        args.difficulty_path,
        args.filter_num,
        args.save_path
    )


if __name__ == "__main__":
    main() 