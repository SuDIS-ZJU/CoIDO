# This file is adapted from the following projects:
# - https://github.com/haotian-liu/LLaVA
# - https://github.com/lm-sys/FastChat
# - https://github.com/tatsu-lab/stanford_alpaca
#
# Original Copyright Notice:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
This file implements the first stage training of the CoIDO model.
Main functionalities include:
1. Data preprocessing and dataset loading
2. Model training configuration
3. Training loop implementation
"""

# Basic library imports
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import wandb

import torch

# Deep learning related library imports
import transformers
import tokenizers
import sys
import os

sys.path.append("./LLaVA")

# Import LLaVA related constants and utilities
from llava.constants import (
    IGNORE_INDEX,  # Index value for masking
    IMAGE_TOKEN_INDEX,  # Image token index
    DEFAULT_IMAGE_TOKEN,  # Default image token
    DEFAULT_IM_START_TOKEN,  # Image start token
    DEFAULT_IM_END_TOKEN,  # Image end token
)
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image

# Import CoIDO specific models and trainers
from coido_model import (
    LlavaConfig_CoIDO,
    LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback,
)
from coido_trainer import LLaVATrainer_CoIDO

local_rank = None

def rank0_print(*args):
    """Print information only on the main process (rank0)"""
    if local_rank == 0:
        print(*args)

from packaging import version

# Check tokenizer version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

@dataclass
class ModelArguments:
    """Model-related configuration parameters"""
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")  # Pre-trained model path
    version: Optional[str] = field(default="v0")  # Model version
    freeze_backbone: bool = field(default=False)  # Whether to freeze the backbone network
    tune_mm_mlp_adapter: bool = field(default=False)  # Whether to fine-tune multimodal MLP adapter
    vision_tower: Optional[str] = field(default=None)  # Vision encoder path
    mm_vision_select_layer: Optional[int] = field(default=-1)  # Layer number for visual features
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)  # Pre-trained multimodal MLP adapter path
    mm_projector_type: Optional[str] = field(default="linear")  # Projector type
    mm_use_im_start_end: bool = field(default=False)  # Whether to use image start/end tokens
    mm_use_im_patch_token: bool = field(default=True)  # Whether to use image patch tokens
    mm_patch_merge_type: Optional[str] = field(default="flat")  # Patch merge method
    mm_vision_select_feature: Optional[str] = field(default="patch")  # Selected visual feature type

@dataclass
class DataArguments:
    """Dataset-related configuration parameters"""
    data_path: str = field(default=None, metadata={"help": "Training data path"})
    lazy_preprocess: bool = False  # Whether to use lazy preprocessing
    is_multimodal: bool = False  # Whether it's multimodal data
    image_folder: Optional[str] = field(default=None)  # Image folder path
    image_aspect_ratio: str = "square"  # Image aspect ratio setting

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training-related configuration parameters"""
    cache_dir: Optional[str] = field(default=None)  # Cache directory
    optim: str = field(default="adamw_torch")  # Optimizer
    remove_unused_columns: bool = field(default=False)  # Whether to remove unused columns
    freeze_mm_mlp_adapter: bool = field(default=False)  # Whether to freeze multimodal MLP adapter
    mpt_attn_impl: Optional[str] = field(default="triton")  # MPT attention implementation
    model_max_length: int = field(  # Maximum sequence length
        default=512,
        metadata={"help": "Maximum sequence length, excess will be truncated or padded."},
    )
    double_quant: bool = field(  # Whether to use double quantization
        default=True,
        metadata={"help": "Whether to compress quantization statistics through double quantization."},
    )
    quant_type: str = field(  # Quantization type
        default="nf4",
        metadata={"help": "Quantization data type, options: 'fp4' or 'nf4'."},
    )
    bits: int = field(default=16, metadata={"help": "Quantization bits"})
    lora_enable: bool = False  # Whether to enable LoRA
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha parameter
    lora_dropout: float = 0.05  # LoRA dropout rate
    lora_weight_path: str = ""  # LoRA weight path
    lora_bias: str = "none"  # LoRA bias setting
    mm_projector_lr: Optional[float] = None  # Learning rate for multimodal projector
    group_by_modality_length: bool = field(default=False)  # Whether to group by modality length
    
    # Spectral clustering related parameters
    use_clustering: bool = field(
        default=False,
        metadata={"help": "Whether to use spectral clustering results for diversity enhancement"},
    )
    clustering_results_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to spectral clustering results file"},
    )

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Process parameters in DeepSpeed ZeRO-3 optimizer
    
    Args:
        param: Parameter to process
        ignore_status: Whether to ignore parameter status check
        name: Parameter name
    Returns:
        Processed parameter copy
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    Get PEFT (Parameter-Efficient Fine-Tuning) model LoRA parameter status
    
    Args:
        named_params: Named parameter iterator
        bias: Bias processing method
    Returns:
        LoRA related parameter status dictionary
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    Get non-LoRA parameter status
    
    Args:
        named_params: Named parameter iterator
        require_grad_only: Whether to return parameters only with gradient
    Returns:
        Non-LoRA parameter status dictionary
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """
    Get multimodal adapter status
    
    Args:
        named_params: Named parameter iterator
        keys_to_match: List of keys to match
    Returns:
        Multimodal adapter parameter status dictionary
    """
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    """
    Find all linear layer names in the model
    
    Args:
        model: Model to search
    Returns:
        List of linear layer names
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Safely save HuggingFace Trainer model
    
    Args:
        trainer: HuggingFace Trainer instance
        output_dir: Output directory
    """
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save adapter
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Adjust tokenizer and embedding size to fit new special tokens
    
    Args:
        special_tokens_dict: Special token dictionary
        tokenizer: tokenizer instance
        model: model instance
    Note: This is unoptimized version, which may result in embedding size not being a multiple of 64
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """
    Tokenize string sequence
    
    Args:
        strings: Sequence of strings to process
        tokenizer: tokenizer instance
    Returns:
        Dictionary containing input_ids, labels, etc.
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    """
    Mask target sequence
    
    Args:
        target: Target sequence
        tokenized_lens: Token length list
        speakers: Speaker list
    """
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """
    Add speaker mark and start/end signal to each dialogue turn
    
    Args:
        header: Dialogue header information
        source: Dialogue source data
        get_conversation: Whether to return complete dialogue
    Returns:
        Processed dialogue text
    """
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """
    Preprocess multimodal data
    
    Args:
        sources: Source data sequence
        data_args: Data parameters
    Returns:
        Processed data
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    """
    Preprocess LLaMA-2 format dialogue data
    
    Args:
        sources: Source data
        tokenizer: tokenizer instance
        has_image: Whether to include image data
    Returns:
        Processed data dictionary
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply dialogue template
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # If the first message is not from human, skip
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize dialogue
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask target
    sep = "### "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"Warning: tokenization length mismatch: {cur_len} vs. {total_len}."
                    f" (Ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    """
    Preprocess v1 format dialogue data
    
    Args:
        sources: Source data
        tokenizer: tokenizer instance
        has_image: Whether to include image data
    Returns:
        Processed data dictionary
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply dialogue template
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # If the first message is not from human, skip
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize dialogue
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask target
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"Warning: tokenization length mismatch: {cur_len} vs. {total_len}."
                    f" (Ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    """
    Preprocess MPT format dialogue data
    
    Args:
        sources: Source data
        tokenizer: tokenizer instance
        has_image: Whether to include image data
    Returns:
        Processed data dictionary
    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply dialogue template
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # If the first message is not from human, skip
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize dialogue
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask target
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"Warning: tokenization length mismatch: {cur_len} vs. {total_len}."
                    f" (Ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocess plain format dialogue data
    
    Args:
        sources: Source data sequence
        tokenizer: tokenizer instance
    Returns:
        Processed data dictionary
    """
    # Add end mark and connect
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # Tokenize dialogue
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Preprocess dialogue data
    
    Processing steps:
    1. Add '### ' mark at the beginning of each sentence, and '\n' at the end
    2. Connect dialogues together
    3. Tokenize connected dialogues
    4. Create a deep copy of target sequence, mask human's words with IGNORE_INDEX
    
    Args:
        sources: Source data sequence
        tokenizer: tokenizer instance
        has_image: Whether to include image data
    Returns:
        Processed data dictionary
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)

    # Add end mark and connect
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # Tokenize dialogue
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """
    Supervised fine-tuning data set class
    
    Implements delayed loading mechanism, processing data only when needed
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Data file path
            tokenizer: tokenizer instance
            data_args: Data parameters
        """
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting input... skipping in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        """Return dataset size"""
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """
        Calculate length of each sample
        
        Returns:
            List of all sample lengths
        """
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        """
        Calculate modality length of each sample
        
        Returns:
            List of all sample modality lengths (positive for image, negative for pure text)
        """
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Get one sample from dataset
        
        Args:
            i: Sample index
        Returns:
            Dictionary containing processed data
        """
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it's wrapped in a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":
                # Expand image to square
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # Process image data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # Data has no image, but model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            
        data_dict['unique_idx']=self.list_data_dict[i]['unique_idx']
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Supervised dataset data collator
    
    Responsible for organizing multiple samples into batch
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Organize instances into batch
        
        Args:
            instances: Instance sequence
        Returns:
            Organized batch data
        """
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        unique_indices=[instance['unique_idx'] for instance in instances]
        batch['unique_indices']=unique_indices
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """
    Create supervised fine-tuning data module
    
    Args:
        tokenizer: tokenizer instance
        data_args: Data parameters
    Returns:
        Dictionary containing train dataset, eval dataset, and data collator
    """
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train(attn_implementation=None):
    """
    Training function entry point
    
    Args:
        attn_implementation: Attention mechanism implementation
    """
    # No wandb
    # os.environ["WANDB_DISABLED"] = "true"
    
    global local_rank

    # Parse command line arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # Initialize wandb (only in main process)
    if local_rank == 0 or local_rank == -1:
        # Build experiment name
        experiment_name = "CoIDO_stage1"
        if training_args.use_clustering:
            experiment_name += "_clustering"
            
        # Initialize wandb project
        wandb.init(
            project="CoIDO",
            name=experiment_name,
            config={
                "model": model_args.model_name_or_path,
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "epochs": training_args.num_train_epochs,
                "use_clustering": training_args.use_clustering,
                "model_version": model_args.version,
                "vision_tower": model_args.vision_tower,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": training_args.per_device_train_batch_size * 
                                      training_args.gradient_accumulation_steps * 
                                      (torch.cuda.device_count() if torch.cuda.is_available() else 1),
                "mm_projector_type": model_args.mm_projector_type,
            },
            tags=[
                f"model_{model_args.version}",
                "clustering" if training_args.use_clustering else "no_clustering",
            ]
        )
        print(f"Wandb initialization successful: {experiment_name}")
    
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Set quantization parameters
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    # Load model
    if model_args.vision_tower is not None:
        if "mpt" in model_args.model_name_or_path:
            raise NotImplementedError
        else:
            model = LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
            )
            # Verify and fix log_sigma parameters
            check_and_fix_log_sigma_params(model, training_args)
    else:
        raise NotImplementedError

    # Set generation configuration
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.9
        model.generation_config.top_p = 0.6

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # Prepare model for quantization training
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    # Set gradient checkpoint
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Configure LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        # Method 1: Directly specify target module names, avoid using automatic detection
        target_modules = [
            "q_proj",   # Attention module query projection
            "k_proj",   # Attention module key projection
            "v_proj",   # Attention module value projection
            "o_proj",   # Attention module output projection
            "gate_proj",  # MLP module gate projection
            "up_proj",    # MLP module upper projection
            "down_proj",  # MLP module lower projection
        ]
        
        # Method 2: Add module type matching
        modules_to_save = None
        
        rank0_print(f"Using specified target modules for LoRA: {target_modules}")
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapter...")
        
        # Optional: Try not to use Flash Attention for LoRA application
        orig_attn_implementation = None
        if hasattr(model.config, "attn_implementation") and model.config.attn_implementation == "flash_attention_2":
            rank0_print("Temporarily disable Flash Attention 2 for LoRA application...")
            orig_attn_implementation = model.config.attn_implementation
            model.config.attn_implementation = "sdpa"  # Use default attention
            
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Restore Flash Attention setting (if modified)
        if orig_attn_implementation is not None:
            rank0_print(f"Restoring attention implementation to: {orig_attn_implementation}")
            model.config.attn_implementation = orig_attn_implementation

    # Load tokenizer
    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # Adjust tokenizer and model
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    # Initialize vision module
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # Configure multimodal MLP adapter
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Process quantization model data type
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Create data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Create trainer
    trainer = LLaVATrainer_CoIDO(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # Save model
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    # Complete wandb record (only in main process)
    if local_rank == 0 or local_rank == -1:
        wandb.finish()


def check_and_fix_log_sigma_params(model, training_args):
    """
    Check and fix log_sigma parameters
    
    Args:
        model: Model instance
        training_args: Training parameters
    """
    # Check log_sigma_1 parameter
    if not hasattr(model, 'log_sigma_1') or model.log_sigma_1.numel() == 0:
        print("Warning: log_sigma_1 parameter does not exist or is empty, reinitializing...")
        model.log_sigma_1 = torch.nn.Parameter(
            torch.tensor([0.0], 
                        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
                        device=training_args.device)
        )
    
    # Check log_sigma_2 parameter
    if not hasattr(model, 'log_sigma_2') or model.log_sigma_2.numel() == 0:
        print("Warning: log_sigma_2 parameter does not exist or is empty, reinitializing...")
        model.log_sigma_2 = torch.nn.Parameter(
            torch.tensor([0.0], 
                        dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
                        device=training_args.device)
        )
    
    # Brief log, confirm parameters are normal
    if hasattr(model, 'log_sigma_1') and hasattr(model, 'log_sigma_2'):
        if model.log_sigma_1.numel() > 0 and model.log_sigma_2.numel() > 0:
            print("Uncertainty weight parameter initialization successful")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
