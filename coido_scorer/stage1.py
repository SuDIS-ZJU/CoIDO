"""COIDO Stage-1: train the multimodal scorer (LLaVA-based)."""
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import datetime
import wandb
import torch
import transformers
import tokenizers
import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
sys.path.append("./LLaVA")
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from PIL import Image
from coido_scorer_model import LlavaConfig_CoIDO, LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback
from coido_trainer import LLaVATrainer_CoIDO
local_rank = None

def rank0_print(*args):
    """(rank0)"""
    if local_rank == 0:
        print(*args)
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')
    version: Optional[str] = field(default='v0')
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default='patch')

@dataclass
class DataArguments:
    """Data configuration arguments."""
    data_path: str = field(default=None, metadata={'help': 'Path to the training data JSON file.'})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Training configuration arguments, extending HuggingFace TrainingArguments."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default='triton')
    model_max_length: int = field(default=512, metadata={'help': 'Maximum sequence length for the model.'})
    double_quant: bool = field(default=True, metadata={'help': 'Whether to use double quantization (for bitsandbytes).'})
    quant_type: str = field(default='nf4', metadata={'help': "'fp4' 'nf4'"})
    bits: int = field(default=16, metadata={'help': 'Number of bits for quantization (4, 8, or 16).'})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    clustering_results_path: Optional[str] = field(default=None, metadata={'help': 'Path to pre-computed clustering results JSON.'})

def maybe_zero_3(param, ignore_status=False, name=None):
    """DeepSpeed ZeRO-3 
 
 Args:
 param: 
 ignore_status: 
 name: 
 Returns"""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, 'ds_id'):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f'{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    """PEFT(Parameter-Efficient Fine-Tuning) LoRA 
 
 Args:
 named_params: 
 bias: 
 Returns:
 LoRA"""
    if bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    elif bias == 'all':
        to_return = {k: t for k, t in named_params if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if 'lora_' in k:
                to_return[k] = t
                bias_name = k.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """LoRA 
 
 Args:
 named_params: 
 require_grad_only: 
 Returns:
 LoRA"""
    to_return = {k: t for k, t in named_params if 'lora_' not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """Args:
 named_params: 
 keys_to_match: 
 Returns"""
    to_return = {k: t for k, t in named_params if any((key_match in k for key_match in keys_to_match))}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    """Args:
 model: 
 Returns"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any((mm_keyword in name for mm_keyword in multimodal_keywords)):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """HuggingFace Trainer 
 
 Args:
 trainer: HuggingFace Trainer 
 output_dir"""
    if getattr(trainer.args, 'tune_mm_mlp_adapter', False):
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, 'use_im_start_end', False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, 'mm_projector')
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
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

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    """tokenizer embedding token
 
 Args:
 special_tokens_dict: token 
 tokenizer: tokenizer 
 model: 
 : , embedding 64"""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """tokenize 
 
 Args:
 strings: 
 tokenizer: tokenizer 
 Returns:
 input_ids,labels"""
    tokenized_list = [tokenizer(text, return_tensors='pt', padding='longest', max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def _mask_targets(target, tokenized_lens, speakers):
    """mask 
 
 Args:
 target: 
 tokenized_lens: token 
 speakers"""
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == 'human':
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Args:
 header: 
 source: 
 get_conversation: 
 Returns"""
    BEGIN_SIGNAL = '### '
    END_SIGNAL = '\n'
    conversation = header
    for sentence in source:
        from_str = sentence['from']
        if from_str.lower() == 'human':
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == 'gpt':
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence['value'] = BEGIN_SIGNAL + from_str + ': ' + sentence['value'] + END_SIGNAL
        if get_conversation:
            conversation += sentence['value']
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """Args:
 sources: 
 data_args: 
 Returns"""
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if 'mmtag' in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool=False) -> Dict:
    """LLaMA-2 
 
 Args:
 sources: 
 tokenizer: tokenizer 
 has_image: 
 Returns"""
    conv = conversation_lib.default_conversation.copy()
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(conversations, return_tensors='pt', padding='longest', max_length=tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2
    sep = '### '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == '':
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
            target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f'tokenization{cur_len} vs. {total_len}( )')
    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool=False) -> Dict:
    """v1 
 
 Args:
 sources: 
 tokenizer: tokenizer 
 has_image: 
 Returns"""
    conv = conversation_lib.default_conversation.copy()
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(conversations, return_tensors='pt', padding='longest', max_length=tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == '':
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
            if i != 0 and (not tokenizer.legacy) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f'tokenization{cur_len} vs. {total_len}( )')
    return dict(input_ids=input_ids, labels=targets)

def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool=False) -> Dict:
    """MPT 
 
 Args:
 sources: 
 tokenizer: tokenizer 
 has_image: 
 Returns"""
    conv = conversation_lib.default_conversation.copy()
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(conversations, return_tensors='pt', padding='longest', max_length=tokenizer.model_max_length, truncation=True).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == '':
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
            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1
            target[cur_len:cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f'tokenization{cur_len} vs. {total_len}( )')
    return dict(input_ids=input_ids, labels=targets)

def preprocess_plain(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Args:
 sources: 
 tokenizer: tokenizer 
 Returns"""
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)

def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool=False) -> Dict:
    """1. '
 2. 
 3. tokenize
 4. , human IGNORE_INDEX mask
 
 Args:
 sources: 
 tokenizer: tokenizer 
 has_image: 
 Returns"""
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith('v1'):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == 'mpt':
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    conversations = []
    for source in sources:
        header = f'{conversation_lib.default_conversation.system}\n\n'
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]
    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized['input_ids']
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s['value'] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s['value'] for s in source], tokenizer)['input_ids_lens']
        speakers = [sentence['from'] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning. Each item is processed on-the-fly."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        """Args:
 data_path: 
 tokenizer: tokenizer 
 data_args"""
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, 'r'))
        rank0_print('Formatting inputs...')
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.skipped_images = []
        self.skipped_count = 0
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.skipped_log_file = f'./logs/skipped_images_{timestamp}.txt'
        os.makedirs(os.path.dirname(self.skipped_log_file), exist_ok=True)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """Returns"""
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum((len(conv['value'].split()) for conv in sample['conversations'])) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """Returns:
 ( , )"""
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum((len(conv['value'].split()) for conv in sample['conversations']))
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _log_skipped_image(self, image_path, reason, sample_index):
        skip_info = {'image_path': image_path, 'reason': reason, 'sample_index': sample_index, 'timestamp': datetime.datetime.now().isoformat()}
        self.skipped_images.append(skip_info)
        self.skipped_count += 1
        with open(self.skipped_log_file, 'a', encoding='utf-8') as f:
            f.write(f'{skip_info['timestamp']} - Sample {sample_index}: {image_path} - {reason}\n')
        print(f'Skipped image {sample_index}: {image_path} - {reason}')

    def get_skipped_summary(self):
        return {'total_skipped': self.skipped_count, 'skipped_images': self.skipped_images, 'log_file': self.skipped_log_file}

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """Args:
 i: 
 Returns"""
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Multiple sources per sample is not supported."
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_path = os.path.join(image_folder, image_file)
            try:
                if not os.path.exists(image_path):
                    self._log_skipped_image(image_path, 'Image not found: ', i)
                    return self.__getitem__((i + 1) % len(self.list_data_dict))
                file_size = os.path.getsize(image_path)
                if file_size < 100:
                    self._log_skipped_image(image_path, f'Image file too small ({file_size} bytes)', i)
                    return self.__getitem__((i + 1) % len(self.list_data_dict))
                image = Image.open(image_path)
                image.verify()
                image = Image.open(image_path).convert('RGB')
            except (OSError, IOError, Image.UnidentifiedImageError) as e:
                self._log_skipped_image(image_path, f'Image error: {str(e)}', i)
                return self.__getitem__((i + 1) % len(self.list_data_dict))
            except Exception as e:
                self._log_skipped_image(image_path, f'Image error: {str(e)}', i)
                return self.__getitem__((i + 1) % len(self.list_data_dict))
            if self.data_args.image_aspect_ratio == 'pad':

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple((int(x * 255) for x in processor.image_mean)))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(copy.deepcopy([e['conversations'] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e['conversations'] for e in sources])
        data_dict = preprocess(sources, self.tokenizer, has_image='image' in self.list_data_dict[i])
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0], labels=data_dict['labels'][0])
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        data_dict['unique_idx'] = self.list_data_dict[i]['unique_idx']
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """batch"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """batch
 
 Args:
 instances: 
 Returns:
 batch"""
        input_ids, labels = tuple(([instance[key] for instance in instances] for key in ('input_ids', 'labels')))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all((x is not None and x.shape == images[0].shape for x in images)):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        unique_indices = [instance['unique_idx'] for instance in instances]
        batch['unique_indices'] = unique_indices
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Args:
 tokenizer: tokenizer 
 data_args: 
 Returns:
 ,"""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train(attn_implementation=None):
    """Args:
 attn_implementation"""
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    if local_rank == 0 or local_rank == -1:
        experiment_name = 'coido_clustering'
        wandb.init(project='COIDO', name=experiment_name, config={
            'model': model_args.model_name_or_path,
            'batch_size': training_args.per_device_train_batch_size,
            'learning_rate': training_args.learning_rate,
            'epochs': training_args.num_train_epochs,
            'model_version': model_args.version,
            'vision_tower': model_args.vision_tower,
            'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
            'effective_batch_size': training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * (torch.cuda.device_count() if torch.cuda.is_available() else 1),
            'mm_projector_type': model_args.mm_projector_type,
        }, tags=['clustering', f'model_{model_args.version}'])
        print(f'Wandb{experiment_name}')
    compute_dtype = torch.float16 if training_args.fp16 else torch.bfloat16 if training_args.bf16 else torch.float32
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(device_map={'': training_args.device}, load_in_4bit=training_args.bits == 4, load_in_8bit=training_args.bits == 8, quantization_config=BitsAndBytesConfig(load_in_4bit=training_args.bits == 4, load_in_8bit=training_args.bits == 8, llm_int8_skip_modules=['mm_projector'], llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=training_args.double_quant, bnb_4bit_quant_type=training_args.quant_type)))
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            raise NotImplementedError
        else:
            model = LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
                **bnb_model_from_pretrained_args)
    else:
        raise NotImplementedError
    if hasattr(model, 'generation_config'):
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.9
        model.generation_config.top_p = 0.6
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = torch.float32 if training_args.fp16 else torch.bfloat16 if training_args.bf16 else torch.float32
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=training_args.lora_r, lora_alpha=training_args.lora_alpha, target_modules=find_all_linear_names(model), lora_dropout=training_args.lora_dropout, bias=training_args.lora_bias, task_type='CAUSAL_LM')
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print('LoRA')
        model = get_peft_model(model, lora_config)
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side='right')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side='right', use_fast=False)
    if model_args.version == 'v0':
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token='[PAD]'), tokenizer=tokenizer, model=model)
    elif model_args.version == 'v0.5':
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates['vicuna_v1']
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer_CoIDO(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if hasattr(trainer.train_dataset, 'get_skipped_summary'):
        skipped_summary = trainer.train_dataset.get_skipped_summary()
        print('\n' + '=' * 60)
        print('Skipped images summary:')
        print('=' * 60)
        print(f'Total skipped images: {skipped_summary["total_skipped"]}')
        print(f'Skipped images log: {skipped_summary["log_file"]}')
        if skipped_summary['total_skipped'] > 0:
            print('Skipped images summary:')
            reason_counts = {}
            for img_info in skipped_summary['skipped_images']:
                reason = img_info['reason'].split(':')[0]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            for reason, count in reason_counts.items():
                print(f'  - {reason}: {count} images')
        print('=' * 60 + '\n')
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if local_rank == 0 or local_rank == -1:
        wandb.finish()

if __name__ == '__main__':
    train(attn_implementation='flash_attention_2')
