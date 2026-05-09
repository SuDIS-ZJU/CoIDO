#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaModel


class LlavaConfig_CoIDO(LlamaConfig):
    model_type = "llava_coido_scorer"


class LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    COIDO,MLP
    
    DeepSpeed ZeRO-3Transformer,
    MLP,
    (5)
    """
    config_class = LlavaConfig_CoIDO

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.hidden_size = 384
        self.intermediate_size = 768
        self.dropout_prob = 0.2

        self.clip_projector = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(512, self.hidden_size)
        )

        self.scores_projector_1d = nn.Sequential(
            nn.Linear(1, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(self.dropout_prob),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(self.dropout_prob),
            nn.Linear(256, self.hidden_size)
        )
        self.scores_projector_3d = nn.Sequential(
            nn.Linear(3, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(self.dropout_prob),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(self.dropout_prob),
            nn.Linear(256, self.hidden_size)
        )
        self.scores_projector_5d = nn.Sequential(
            nn.Linear(5, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(self.dropout_prob),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(self.dropout_prob),
            nn.Linear(256, self.hidden_size)
        )

        
        self.fusion_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout_prob/2),
            nn.Linear(128, 1)
        )

        self.s1 = nn.Parameter(torch.tensor(0.0))
        self.s2 = nn.Parameter(torch.tensor(0.0))
        print(f" s1: {self.s1.shape}, s2: {self.s2.shape}")

        self.post_init()
        print("\n(MLP)...")

    def get_model(self):
        return self.model

    def get_score_net_dtype(self):
        return next(self.clip_projector.parameters()).dtype

    def predict_weights(self, combined_features):
        """
        MLP,
        : Project -> MLP -> MLP -> Classify
        """
        clip_features = combined_features[:, :1536]
        score_features_input = combined_features[:, 1536:]
        score_features_dim = score_features_input.shape[1]

        clip_proj = self.clip_projector(clip_features) # [B, H]

        if score_features_dim == 1:
            scores_proj = self.scores_projector_1d(score_features_input) # [B, H]
        elif score_features_dim == 3:
            scores_proj = self.scores_projector_3d(score_features_input) # [B, H]
        elif score_features_dim == 5:
            scores_proj = self.scores_projector_5d(score_features_input) # [B, H]
        else:
            print(f": {score_features_dim}. ")
            target_dim = 1
            target_projector = self.scores_projector_1d
            if score_features_dim > 3:
                 target_dim = 5
                 target_projector = self.scores_projector_5d
            elif score_features_dim > 1:
                 target_dim = 3
                 target_projector = self.scores_projector_3d

            batch_size = score_features_input.shape[0]
            adjusted_scores = torch.zeros((batch_size, target_dim),
                                          device=score_features_input.device,
                                          dtype=score_features_input.dtype)
            copy_dim = min(score_features_dim, target_dim)
            adjusted_scores[:, :copy_dim] = score_features_input[:, :copy_dim]
            scores_proj = target_projector(adjusted_scores) # [B, H]

        fused_input = torch.cat([clip_proj, scores_proj], dim=1) # [B, H*2]

        fusion_output = self.fusion_network(fused_input)

        enhanced_output = fusion_output + self.feature_enhancer(fusion_output)

        weights = self.classifier(enhanced_output) # [B, 1]

        return weights

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes,
            )
        return super().forward(
             input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
             inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions,
             output_hidden_states=output_hidden_states, return_dict=return_dict,
         )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask, None, None, images, image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoModelForCausalLM.register(LlavaConfig_CoIDO, LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback)
