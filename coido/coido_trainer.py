# This file is adapted from https://github.com/haotian-liu/LLaVA

"""
CoIDO trainer implementation.
Main functionalities:
1. Implement weighted loss function computation
2. Support multimodal data processing
3. Implement distributed training sampler
4. Provide optimizer configuration and checkpoint saving
"""

import os
import torch
import torch.nn as nn

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
    Process parameters in DeepSpeed ZeRO-3 optimizer
    
    Args:
        param: Parameter to process
        ignore_status: Whether to ignore status check
        name: Parameter name
    Returns:
        Processed parameter copy
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
    Get multimodal adapter state, supporting ZeRO-3 optimizer
    
    Args:
        named_params: Named parameter iterator
        keys_to_match: List of keys to match
    Returns:
        Multimodal adapter parameter state dictionary
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
    Split index list into chunks of approximately equal length
    
    Args:
        indices: Index list
        lengths: Length corresponding to each index
        num_chunks: Number of chunks to split into
    Returns:
        List of split chunks
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
    Get indices grouped by modality and length
    
    Args:
        lengths: Sample length list (positive for multimodal samples, negative for text-only samples)
        batch_size: Batch size
        world_size: Number of processes in distributed training
        generator: Random number generator
    Returns:
        Grouped index list
    """
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # All samples are of the same modality
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    # Separate multimodal and text-only samples
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # Perform length grouping for each modality separately
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
    
    # Combine grouped indices into megabatches
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    # Handle last incomplete batch
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    
    # Randomly shuffle megabatch order
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    """
    Get length-grouped indices
    
    Args:
        lengths: Sample length list
        batch_size: Batch size
        world_size: Number of processes in distributed training
        generator: Random number generator
        merge: Whether to merge batches
    Returns:
        Grouped index list
    """
    # Randomly shuffle indices
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    
    # Split indices into megabatches
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    
    # Sort by length within each megabatch
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    
    # Split each megabatch into world_size sub-batches
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """
    Length-grouped sampler
    
    Group data points with similar feature lengths together, while maintaining some randomness
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
        Initialize sampler
        
        Args:
            batch_size: Batch size
            world_size: Number of processes in distributed training
            lengths: Sample length list
            generator: Random number generator
            group_by_modality: Whether to group by modality
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
        Return grouped index iterator
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
    CoIDO specific LLaVA trainer
    
    Added weighted loss computation and multimodal data processing support
    """
    def __init__(self, **kwargs):
        """
        Initialize trainer
        
        Args:
            **kwargs: Other trainer parameters
        """
        super().__init__(**kwargs)
        # Since we simplified to only use clip+scores, set it as fixed
        self.feature_extractor_setting = "clip+scores"
        
        # Initialize uncertainty parameters as part of the model
        underlying_model = self._get_underlying_model(self.model)
        if not hasattr(underlying_model, 'log_sigma1'):
            underlying_model.log_sigma1 = nn.Parameter(torch.zeros(1))
            underlying_model.log_sigma2 = nn.Parameter(torch.zeros(1))
        
        # Load CLIP features and precomputed scores
        if self.feature_extractor_setting == "clip" or self.feature_extractor_setting == "clip+scores":
            self.clip_feat = torch.load("./data/scores/llava_clip_feature.pt")
        if self.feature_extractor_setting == "scores" or self.feature_extractor_setting == "clip+scores":
            score_names = [
                "./data/scores/llava_clipscore.json",
                "./data/scores/llava_imagereward.json",
                "./data/scores/deepseek-chat/processed_score.json",
            ]
            self.score_dicts = self._load_scores(score_names)
            
        # Load clustering results (if exists)
        self.use_clustering = getattr(self.args, "use_clustering", False)
        self.clustering_gamma = getattr(self.args, "clustering_gamma", 0.1)
        self.clustering_results = None
        
        if self.use_clustering and hasattr(self.args, "clustering_results_path"):
            self._load_clustering_results(self.args.clustering_results_path)

    def _load_clustering_results(self, clustering_results_path):
        """
        Load clustering results
        
        Args:
            clustering_results_path: Path to clustering results file
        """
        if clustering_results_path and os.path.exists(clustering_results_path):
            print(f"Loading clustering results: {clustering_results_path}")
            with open(clustering_results_path, "r") as f:
                clustering_data = json.load(f)
                
            if "cluster_labels" in clustering_data:
                # New format: cluster_labels is a list of labels
                cluster_labels = np.array(clustering_data["cluster_labels"])
                sample_indices = clustering_data.get("sample_indices", None)
                
                # Create sample ID to cluster label mapping
                self.sample_to_cluster = {}
                
                if sample_indices is not None:
                    # If sampling is used, sample indices need to be used
                    for i, cluster_id in enumerate(cluster_labels):
                        sample_id = str(sample_indices[i])
                        self.sample_to_cluster[sample_id] = int(cluster_id)
                else:
                    # If sampling is not used, sample ID is index
                    for i, cluster_id in enumerate(cluster_labels):
                        self.sample_to_cluster[str(i)] = int(cluster_id)
                        
                # Create cluster ID to samples mapping
                self.cluster_to_samples = {}
                for sample_id, cluster_id in self.sample_to_cluster.items():
                    if cluster_id not in self.cluster_to_samples:
                        self.cluster_to_samples[cluster_id] = []
                    self.cluster_to_samples[cluster_id].append(sample_id)
                
                print(f"Loaded clustering information for {len(self.sample_to_cluster)} samples")
                print(f"Cluster count: {len(self.cluster_to_samples)}")
            else:
                # Old format: directly cluster to sample mapping
                self.cluster_to_samples = clustering_data
                
                # Create sample ID to cluster label mapping
                self.sample_to_cluster = {}
                for cluster_id, samples in self.cluster_to_samples.items():
                    for sample_id in samples:
                        self.sample_to_cluster[sample_id] = int(cluster_id)
                        
                print(f"Loaded clustering information for {len(self.sample_to_cluster)} samples")
                print(f"Cluster count: {len(self.cluster_to_samples)}")
            
            self.clustering_results = clustering_data
        else:
            print(f"Clustering results file does not exist: {clustering_results_path}")
            self.use_clustering = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Get training data sampler
        
        If multimodal grouping is enabled, use LengthGroupedSampler
        Otherwise use parent class default sampler
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
        Create optimizer, ensuring uncertainty parameters are included
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # If multimodal projector learning rate is set
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name
                ]
                # Set different optimization configurations for different parameter groups
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and "log_sigma" not in n
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
                                and "log_sigma" not in n
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
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if "log_sigma" in n and p.requires_grad
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                # Default parameter grouping
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and "log_sigma" not in n and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and "log_sigma" not in n and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if "log_sigma" in n and p.requires_grad
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
            
            # Handle 8-bit Adam optimizer
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
        Save checkpoint
        
        If only tuning multimodal adapter, only save adapter weights
        Otherwise save full model checkpoint
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save adapter
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
        Save model
        
        If only tuning multimodal adapter, skip saving
        Otherwise execute normal saving process
        """
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer_CoIDO, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss function
        
        Implements weighted loss computation, supports multimodal input processing
        
        Args:
            model: Model instance
            inputs: Input data dictionary
            return_outputs: Whether to return model outputs
        Returns:
            loss: Calculated loss value
            outputs: (Optional) Model outputs
        """
        unique_indices = inputs.pop("unique_indices")

        assert "inputs_embeds" not in inputs
        images = inputs["images"]
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]
        past_key_values = inputs.get("past_key_values")
        position_ids = inputs.get("position_ids")

        # Preprocess multimodal inputs
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self._get_underlying_model(model).prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels, images
        )

        inputs["input_ids"] = input_ids
        inputs["position_ids"] = position_ids
        inputs["attention_mask"] = attention_mask
        inputs["past_key_values"] = past_key_values
        inputs["inputs_embeds"] = inputs_embeds
        inputs["labels"] = labels

        # Compute base loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        labels = inputs["labels"]
        logits = outputs.logits

        # Compute weighted loss
        loss = self._get_weighted_loss(model, unique_indices, logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _get_underlying_model(self, model):
        """Get underlying model, handling DeepSpeed packaging"""
        if hasattr(model, 'module'):
            return model.module
        return model

    def _load_scores(self, score_names: List[str]):
        """
        Load and normalize precomputed scores
        
        Args:
            score_names: List of score file paths
        Returns:
            Normalized score dictionary list
        """
        def norm_scores(score_dict: dict):
            """
            Normalize scores, adjusting range to [-1,1]
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
            with open(score_name, "r") as f:
                score_dict = json.load(f)
                score_dicts.append(norm_scores(score_dict))

        return score_dicts

    def _get_weighted_loss(self, model, unique_indices, logits, labels):
        """
        Compute weighted loss with uncertainty
        
        Use learnable uncertainty parameters σ1 and σ2 to weight loss
        L = (1/σ1^2) * L1 + (1/σ2^2) * L2 + 2log σ1 + 2log σ2
        
        Args:
            model: Model instance
            unique_indices: Unique index of sample
            logits: Model output logits
            labels: True labels
        Returns:
            Weighted loss value
        """
        # Compute base cross-entropy loss L1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(shift_logits.device)
        
        bsz, max_seq_len, vocab_size = shift_logits.shape
        shift_probs = torch.softmax(shift_logits.view(-1, vocab_size), dim=-1)
        shift_labels = shift_labels.view(-1)
        
        valid_mask = shift_labels > -10
        shift_labels = shift_labels * valid_mask
        
        L1 = torch.sum(
            -torch.log(shift_probs[range(0, shift_labels.shape[0]), shift_labels])
            * valid_mask
        ) / torch.sum(valid_mask)

        # Compute diversity loss L2
        if self.use_clustering and self.clustering_results is not None:
            weights = self._get_batch_weight(model, unique_indices)
            L2 = self._compute_clustering_diversity_loss(unique_indices, weights)
        else:
            L2 = torch.tensor(0.0, device=L1.device)
            
        # Compute total loss with uncertainty weighting
        underlying_model = self._get_underlying_model(model)
        sigma1 = torch.exp(underlying_model.log_sigma1)
        sigma2 = torch.exp(underlying_model.log_sigma2)
        
        # Ensure all tensors are scalars
        L1 = L1.mean()  # If L1 is not scalar
        L2 = L2.mean()  # If L2 is not scalar
        sigma1 = sigma1.mean()  # Ensure sigma1 is scalar
        sigma2 = sigma2.mean()  # Ensure sigma2 is scalar
        
        total_loss = (1.0 / (2.0 * sigma1 ** 2)) * L1 + \
                     (1.0 / (2.0 * sigma2 ** 2)) * L2 + \
                     torch.log(sigma1) + \
                     torch.log(sigma2)
        
        # Record current sigma value for monitoring
        if self.args.local_rank == 0:  # Record only on main process
            if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
                step = self.state.global_step
                if step % 100 == 0:  # Record every 100 steps
                    logger.info(f"Step {step}: sigma1 = {sigma1.item():.4f}, sigma2 = {sigma2.item():.4f}")
                    logger.info(f"Step {step}: L1 = {L1.item():.4f}, L2 = {L2.item():.4f}")
        
        return total_loss
    
    def _compute_clustering_diversity_loss(self, unique_indices, weights):
        """
        Compute clustering-based diversity loss
        
        Main goal is to balance average loss of each cluster, ensuring diversity
        
        Args:
            unique_indices: Unique index of batch samples
            weights: Sample weights
        Returns:
            Diversity loss value
        """
        if not hasattr(self, "sample_to_cluster") or len(self.sample_to_cluster) == 0:
            return torch.tensor(0.0, device=weights.device)
        
        # Get cluster each sample belongs to
        batch_clusters = []
        valid_indices = []
        
        for i, unique_idx in enumerate(unique_indices):
            unique_idx_str = str(unique_idx)
            if unique_idx_str in self.sample_to_cluster:
                batch_clusters.append(self.sample_to_cluster[unique_idx_str])
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=weights.device)
        
        # Compute average weight of each cluster
        cluster_weights = {}
        for i, cluster_id in enumerate(batch_clusters):
            if cluster_id not in cluster_weights:
                cluster_weights[cluster_id] = []
            cluster_weights[cluster_id].append(weights[valid_indices[i]])
        
        # Compute cluster average weights
        cluster_avg_weights = []
        for cluster_id, w_list in cluster_weights.items():
            if len(w_list) > 0:
                avg_weight = torch.stack(w_list).mean()
                cluster_avg_weights.append(avg_weight)
        
        if len(cluster_avg_weights) <= 1:
            return torch.tensor(0.0, device=weights.device)
        
        # Compute cluster weight variance as diversity loss
        # Minimize variance to balance weights of each cluster
        cluster_avg_weights = torch.stack(cluster_avg_weights)
        diversity_loss = torch.var(cluster_avg_weights)
        
        return diversity_loss

    def _get_batch_weight(self, model, unique_indices):
        """
        Get batch weights
        
        Get corresponding features or scores based on feature extractor setting and use model to predict weights
        
        Args:
            model: Model instance
            unique_indices: Unique index of sample
        Returns:
            Predicted weights
        """
        if self.feature_extractor_setting=='clip':
            underlying_model = self._get_underlying_model(model)
            dtype = underlying_model.get_score_net_dtype()
            scores = (
                torch.stack(
                    [self.clip_feat[str(unique_idx)] for unique_idx in unique_indices],
                    dim=0,
                )
                .cuda()
                .to(dtype=dtype)
            )
        elif self.feature_extractor_setting=='scores':
            scores = [
                [score_dict[str(unique_idx)] for score_dict in self.score_dicts]
                for unique_idx in unique_indices
            ]
            scores = torch.tensor(scores).cuda().bfloat16()
        elif self.feature_extractor_setting=='clip+scores':
            # Get CLIP features
            underlying_model = self._get_underlying_model(model)
            dtype = underlying_model.get_score_net_dtype()
            clip_features = (
                torch.stack(
                    [self.clip_feat[str(unique_idx)] for unique_idx in unique_indices],
                    dim=0,
                )
                .cuda()
                .to(dtype=dtype)
            )
            
            # Get precomputed scores
            score_features = []
            for unique_idx in unique_indices:
                if str(unique_idx) == '51052':
                    # For problem samples, use zero vector instead
                    score_features.append([0.0] * len(self.score_dicts))
                    print(f"Warning: Skipping problem sample index={unique_idx}")
                    continue
                score_features.append([score_dict[str(unique_idx)] for score_dict in self.score_dicts])
            score_features = torch.tensor(score_features).cuda().to(dtype=dtype)
            
            # Combine features
            scores = torch.cat([clip_features, score_features], dim=1)
        else:
            raise NotImplementedError
            
        underlying_model = self._get_underlying_model(model)
        weights = torch.softmax(underlying_model.predict_weights(scores), dim=0)
        return weights

    def train(self, *args, **kwargs):
        """
        Override train function, record sigma value at training end
        """
        result = super().train(*args, **kwargs)
        
        # Record sigma value at training end
        if self.args.local_rank == 0:  # Record only on main process
            underlying_model = self._get_underlying_model(self.model)
            sigma1 = torch.exp(underlying_model.log_sigma1.mean()).item()
            sigma2 = torch.exp(underlying_model.log_sigma2.mean()).item()
            logger.info("=" * 50)
            logger.info("Training end, final uncertainty parameter values:")
            logger.info(f"σ1 (uncertainty of cross-entropy loss) = {sigma1:.4f}")
            logger.info(f"σ2 (uncertainty of diversity loss) = {sigma2:.4f}")
            logger.info(f"1/σ1² (weight of cross-entropy loss) = {1/(sigma1**2):.4f}")
            logger.info(f"1/σ2² (weight of diversity loss) = {1/(sigma2**2):.4f}")
            logger.info("=" * 50)
            
            # Save sigma value to file
            import json
            import os
            
            output_dir = self.args.output_dir
            sigma_file = os.path.join(output_dir, "uncertainty_weights.json")
            sigma_info = {
                "sigma1": sigma1,
                "sigma2": sigma2,
                "weight1": 1/(sigma1**2),
                "weight2": 1/(sigma2**2)
            }
            
            with open(sigma_file, "w") as f:
                json.dump(sigma_info, f, indent=2)
            logger.info(f"Uncertainty parameters saved to: {sigma_file}")
            
        return result
