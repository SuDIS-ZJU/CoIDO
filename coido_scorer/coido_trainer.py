
"""
COIDO
:
1. 
2. 
3. 
4. 
"""

import os
import torch
import torch.nn as nn
import wandb

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional, Dict
import json
import numpy as np


def maybe_zero_3(param, ignore_status=False, name=None):
    """
    DeepSpeed ZeRO-3
    
    Args:
        param: 
        ignore_status: 
        name: 
    Returns:
        
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    ,ZeRO-3
    
    Args:
        named_params: 
        keys_to_match: 
    Returns:
        
    """
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    chunks
    
    Args:
        indices: 
        lengths: 
        num_chunks: chunk
    Returns:
        chunks
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    """
    
    
    Args:
        lengths: (,)
        batch_size: 
        world_size: 
        generator: 
    Returns:
        
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    """
    
    
    Args:
        lengths: 
        batch_size: 
        world_size: 
        generator: 
        merge: 
    Returns:
        
    """
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """
    
    
    ,
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        """
        
        
        Args:
            batch_size: 
            world_size: 
            lengths: 
            generator: 
            group_by_modality: 
        """
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality



    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        """
        
        """
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class LLaVATrainer_CoIDO(Trainer):
    """
    COIDOLLaVA
    
    
    """
    def __init__(self, **kwargs):
        """
        
        Args:
            **kwargs: 
        """
        super().__init__(**kwargs)

        self.clip_feat = torch.load("./data/scores/llava_clip_feature.pt")
        score_names = [
            "./data/scores/llava_clipscore.json",
            "./data/scores/llava_imagereward.json",
            "./data/scores/deepseek-chat_single/processed_score.json",
        ]
        self.score_dicts = self._load_scores(score_names)
            
        self.clustering_results = None
        
        if hasattr(self.args, "clustering_results_path") and self.args.clustering_results_path:
            self._load_clustering_results(self.args.clustering_results_path)

    def _load_clustering_results(self, clustering_results_path):
        """
        
        
        Args:
            clustering_results_path: 
        """
        if clustering_results_path and os.path.exists(clustering_results_path):
            print(f": {clustering_results_path}")
            with open(clustering_results_path, "r") as f:
                clustering_data = json.load(f)
                
            if "cluster_labels" in clustering_data:
                cluster_labels = np.array(clustering_data["cluster_labels"])
                sample_indices = clustering_data.get("sample_indices", None)
                
                self.sample_to_cluster = {}
                
                if sample_indices is not None:
                    for i, cluster_id in enumerate(cluster_labels):
                        sample_id = str(sample_indices[i])
                        self.sample_to_cluster[sample_id] = int(cluster_id)
                else:
                    for i, cluster_id in enumerate(cluster_labels):
                        self.sample_to_cluster[str(i)] = int(cluster_id)
                        
                self.cluster_to_samples = {}
                for sample_id, cluster_id in self.sample_to_cluster.items():
                    if cluster_id not in self.cluster_to_samples:
                        self.cluster_to_samples[cluster_id] = []
                    self.cluster_to_samples[cluster_id].append(sample_id)
                
                print(f" {len(self.sample_to_cluster)} ")
                print(f": {len(self.cluster_to_samples)}")
            else:
                self.cluster_to_samples = {}
                
                self.sample_to_cluster = {}
                for cluster_id, samples in clustering_data.items():
                    if cluster_id.startswith('_') or not isinstance(samples, list):
                        continue
                    
                    self.cluster_to_samples[cluster_id] = samples
                    for sample_id in samples:
                        self.sample_to_cluster[sample_id] = int(cluster_id)
                        
                print(f" {len(self.sample_to_cluster)} ")
                print(f": {len(self.cluster_to_samples)}")
            
            self.clustering_results = clustering_data
        else:
            print(f": {clustering_results_path}")

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        
        
        ,LengthGroupedSampler
        
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        
        
        :
        1. 
        2. 
        3. 8Adam
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        
        
        ,
        
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        else:
            super(LLaVATrainer_CoIDO, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        
        
        ,
        
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer_CoIDO, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        
        
        ,
        
        Args:
            model: 
            inputs: 
            return_outputs: 
        Returns:
            loss: 
            outputs: ()
        """
        unique_indices = inputs.pop("unique_indices")

        assert "inputs_embeds" not in inputs
        images = inputs["images"]
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        past_key_values = inputs.get("past_key_values")
        position_ids = inputs.get("position_ids")

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels, images
        )

        inputs["input_ids"] = input_ids
        inputs["position_ids"] = position_ids
        inputs["attention_mask"] = attention_mask
        inputs["past_key_values"] = past_key_values
        inputs["inputs_embeds"] = inputs_embeds
        inputs["labels"] = labels

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        labels = inputs["labels"]
        logits = outputs.logits

        loss = self._get_weighted_loss(model, unique_indices, logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _load_scores(self, score_names: List[str]):
        """
        
        
        Args:
            score_names: 
        Returns:
            
        """
        def norm_scores(score_dict: dict):
            """
            ,[-1,1]
            """
            min_score = min(score_dict.values())
            max_score = max(score_dict.values())
            normed_score_dict = {
                i[0]: (i[1] - min_score) / (max_score - min_score) * 2 - 1
                for i in score_dict.items()
            }
            return normed_score_dict

        score_dicts = []
        for score_name in score_names:
            print(f": {score_name}")
            with open(score_name, "r") as f:
                score_dict = json.load(f)
                score_dicts.append(norm_scores(score_dict))

        return score_dicts

    def _get_weighted_loss(self, model, unique_indices, logits, labels):
        """
        
        
        ,
        
        
        Args:
            model: 
            unique_indices: 
            logits: logits
            labels: 
        Returns:
            
        """
        orig_weights = self._get_batch_weight(model, unique_indices)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_labels = shift_labels.to(shift_logits.device)

        bsz, max_seq_len, vocab_size = shift_logits.shape
        
        weights_expanded = orig_weights.expand(-1, max_seq_len).reshape(-1) * bsz

        shift_probs = torch.softmax(shift_logits.view(-1, vocab_size), dim=-1)
        shift_labels = shift_labels.view(-1)

        valid_mask = shift_labels > -10

        shift_labels = shift_labels * valid_mask

        weighted_loss = torch.sum(
            -torch.log(shift_probs[range(0, shift_labels.shape[0]), shift_labels])
            * valid_mask
            * weights_expanded
        ) / torch.sum(valid_mask)
        
        w_mean = orig_weights.mean().item()
        w_max = orig_weights.max().item()
        w_min = orig_weights.min().item()
        w_std = orig_weights.std().item()
        
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            if self.optimizer and len(self.optimizer.param_groups) > 0:
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.args.learning_rate
                
            wandb.log({
                "train/weighted_loss": weighted_loss.item(),
                "train/lr": current_lr,
                "train/epoch": self.state.epoch,
                "train/step": self.state.global_step,
                "weights/mean": w_mean,
                "weights/max": w_max,
                "weights/min": w_min,
                "weights/std": w_std,
                "batch/size": bsz,
                "batch/seq_len": max_seq_len
            }, step=self.state.global_step)
        
        if self.clustering_results is not None:
            diversity_loss = self._compute_clustering_diversity_loss(unique_indices, orig_weights)
            
            log_sigma_1 = model.log_sigma_1
            log_sigma_2 = model.log_sigma_2
            
            if log_sigma_1.numel() == 0 or log_sigma_2.numel() == 0:
                device = weighted_loss.device
                dtype = weighted_loss.dtype
                log_sigma_1 = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                log_sigma_2 = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
            
            if weighted_loss.dim() > 0:
                weighted_loss = weighted_loss.mean()
            if diversity_loss.dim() > 0:
                diversity_loss = diversity_loss.mean()
            
            sigma_1 = torch.exp(log_sigma_1)
            sigma_2 = torch.exp(log_sigma_2)
            
            sigma_1_sq = sigma_1 * sigma_1
            sigma_2_sq = sigma_2 * sigma_2
            
            loss_term_1 = weighted_loss / sigma_1_sq
            loss_term_2 = diversity_loss / (2 * sigma_2_sq)
            reg_term = log_sigma_1 + log_sigma_2
            
            total_loss = loss_term_1 + loss_term_2 + reg_term
            
            if total_loss.dim() > 0:
                total_loss = total_loss.mean()
            
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                try:
                    weight_ratio = (sigma_2 / sigma_1).item() if sigma_1.item() > 0 else float('inf')
                    loss_ratio = (loss_term_1 / loss_term_2).item() if loss_term_2.item() != 0 else float('inf')
                    
                    wandb.log({
                        "train/diversity_loss": diversity_loss.item(),
                        "train/task_loss": weighted_loss.item(),
                        "uncertainty/sigma_1": sigma_1.item(),
                        "uncertainty/sigma_2": sigma_2.item(),
                        "uncertainty/loss_term_1": loss_term_1.item(),
                        "uncertainty/loss_term_2": loss_term_2.item(),
                        "uncertainty/reg_term": reg_term.item(),
                        "train/total_loss": total_loss.item(),
                        "uncertainty/loss_ratio": loss_ratio,
                        "uncertainty/weight_ratio": weight_ratio,
                        "clusters/num_in_batch": self._get_num_clusters_in_batch(unique_indices)
                    }, step=self.state.global_step)
                except Exception as e:
                    print(f"wandb: {e}")
            
            return total_loss
    
    def _compute_clustering_diversity_loss(self, unique_indices, weights):
        """
        
        
        ,
        
        Args:
            unique_indices: 
            weights:  ([batch_size, 1])
        Returns:
            
        """
        if not hasattr(self, "sample_to_cluster") or len(self.sample_to_cluster) == 0:
            return torch.tensor(0.0, device=weights.device)
        
        if weights.dim() > 1:
            weights_flat = weights.squeeze()
            if weights_flat.dim() > 1:
                weights_flat = weights_flat[:, 0]
        else:
            weights_flat = weights
            
        batch_clusters = []
        valid_indices = []
        
        for i, unique_idx in enumerate(unique_indices):
            unique_idx_str = str(unique_idx)
            if unique_idx_str in self.sample_to_cluster:
                batch_clusters.append(self.sample_to_cluster[unique_idx_str])
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=weights.device)
        
        cluster_weights_dict = {}
        for i, cluster_id in enumerate(batch_clusters):
            if cluster_id not in cluster_weights_dict:
                cluster_weights_dict[cluster_id] = []
            if i < len(valid_indices):
                idx = valid_indices[i]
                if idx < len(weights_flat):
                    w = weights_flat[idx]
                    cluster_weights_dict[cluster_id].append(w)
        
        cluster_avg_weights = []
        for cluster_id, w_list in cluster_weights_dict.items():
            if len(w_list) > 0:
                try:
                    w_tensor = torch.stack(w_list)
                    avg_weight = torch.mean(w_tensor)
                    cluster_avg_weights.append(avg_weight)
                except Exception as e:
                    pass
        
        if len(cluster_avg_weights) <= 1:
            return torch.tensor(1e-6, device=weights.device, requires_grad=True)
        
        try:
            cluster_avg_weights_tensor = torch.stack(cluster_avg_weights)
            
            mean_weight = torch.mean(cluster_avg_weights_tensor)
            var_weight = torch.mean((cluster_avg_weights_tensor - mean_weight) ** 2)
            std_dev = torch.sqrt(var_weight + 1e-8)
            
            epsilon = 1e-6
            if std_dev < epsilon:
                ideal_weight = torch.mean(cluster_avg_weights_tensor)
                l1_deviation = torch.mean(torch.abs(cluster_avg_weights_tensor - ideal_weight))
                
                if l1_deviation < epsilon:
                    l1_deviation = torch.tensor(epsilon, device=weights.device, requires_grad=True)
                
                return l1_deviation
            
            return std_dev
        except Exception as e:
            return torch.tensor(1e-6, device=weights.device, requires_grad=True)

    def _get_batch_weight(self, model, unique_indices):
        """
        
        
        ,

        Args:
            model: 
            unique_indices: 
        Returns:
            
        """
        dtype = model.get_score_net_dtype()
        clip_features = (
            torch.stack(
                [self.clip_feat[str(unique_idx)] for unique_idx in unique_indices],
                dim=0,
            )
            .cuda()
            .to(dtype=dtype)
        )
        
        score_features = []
        for unique_idx in unique_indices:
            score_features.append([score_dict[str(unique_idx)] for score_dict in self.score_dicts])
        score_features = torch.tensor(score_features).cuda().to(dtype=dtype)
        
        scores = torch.cat([clip_features, score_features], dim=1)
        
        raw_weights = model.predict_weights(scores)
        
        weights = torch.softmax(raw_weights, dim=0)
        return weights

    def _get_num_clusters_in_batch(self, unique_indices):
        """"""
        if not hasattr(self, "sample_to_cluster") or len(self.sample_to_cluster) == 0:
            return 0
            
        clusters_in_batch = set()
        for idx in unique_indices:
            idx_str = str(idx)
            if idx_str in self.sample_to_cluster:
                clusters_in_batch.add(self.sample_to_cluster[idx_str])
                
        return len(clusters_in_batch)
    
    def on_epoch_end(self, args=None, state=None, control=None, **kwargs):
        """
        epoch
        """
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            epoch = self.state.epoch
            
            wandb.log({
                "epoch": epoch,
                "progress/epoch": epoch,
                "progress/total_steps": self.state.global_step,
                "progress/percent_complete": (epoch / self.args.num_train_epochs) * 100
            })
            
            if torch.cuda.is_available():
                memory_stats = {}
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                    memory_stats[f"gpu_{i}/memory_allocated_GB"] = memory_allocated
                    memory_stats[f"gpu_{i}/memory_reserved_GB"] = memory_reserved
                
                memory_stats["gpu/total_memory_allocated_GB"] = sum(
                    torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
                ) / (1024 ** 3)
                
                wandb.log(memory_stats)
            
            if self.clustering_results is not None:
                try:
                    cluster_sizes = {}
                    for cluster_id, samples in self.cluster_to_samples.items():
                        cluster_sizes[cluster_id] = len(samples)
                    
                    wandb.log({
                        "clusters/total_count": len(self.cluster_to_samples),
                        "clusters/min_size": min(cluster_sizes.values()),
                        "clusters/max_size": max(cluster_sizes.values()),
                        "clusters/avg_size": sum(cluster_sizes.values()) / len(cluster_sizes)
                    })
                    
                    wandb.log({
                        "clusters/size_distribution": wandb.Histogram(
                            np.array(list(cluster_sizes.values()))
                        )
                    })
                except Exception as e:
                    print(f": {e}")
        
        return super().on_epoch_end(args, state, control, **kwargs)
