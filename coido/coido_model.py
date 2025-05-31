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
    model_type = "llava_coido"


class LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    CoIDO model fallback solution using MLP instead of Transformer
    
    Specifically designed to solve DeepSpeed ZeRO-3 compatibility issues with Transformer,
    using deep MLP layers for feature fusion instead of Transformer layers to ensure full compatibility with ZeRO-3
    """
    config_class = LlavaConfig_CoIDO

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Feature dimension settings
        self.hidden_size = 256
        
        # CLIP feature processing
        self.clip_projector = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.hidden_size)
        )
        
        # Scores feature processing
        self.scores_projector = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.hidden_size)
        )
        
        # Feature fusion network - using cross-attention style MLP
        self.fusion_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Feature enhancement network - improve model expressiveness
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Add learnable uncertainty weight parameters
        self.log_sigma_1 = nn.Parameter(torch.tensor([0.0]))  # Uncertainty parameter for task 1 loss
        self.log_sigma_2 = nn.Parameter(torch.tensor([0.0]))  # Uncertainty parameter for task 2 loss

        # Print parameter shapes for debugging
        print(f"Initialize log_sigma_1: {self.log_sigma_1.shape}, log_sigma_2: {self.log_sigma_2.shape}")

        # Initialize weights
        self.post_init()
        
        # Print initialization information
        print("\nInitialize fallback MLP weight prediction network...")
        
        try:
            # Print key layer information
            if hasattr(self.fusion_network, "modules"):
                n_params = sum(p.numel() for p in self.fusion_network.parameters() if p.requires_grad)
                print(f"Fusion network trainable parameters: {n_params}")
                
            final_layer = self.classifier[-1]
            if hasattr(final_layer, "weight"):
                final_weight = final_layer.weight.data
                if final_weight.numel() > 0:
                    mean_val = final_weight.mean().item()
                    std_val = final_weight.std().item() if final_weight.numel() > 1 else 0.0
                    print(f"Classifier weight statistics: mean={mean_val:.6f}, std={std_val:.6f}")
            
            # Test random input
            with torch.no_grad():
                batch_size = 2
                sample_clip = torch.randn(batch_size, 1536)
                sample_scores = torch.randn(batch_size, 3)
                sample_input = torch.cat([sample_clip, sample_scores], dim=1)
                
                sample_output = self.predict_weights(sample_input)
                print(f"Random input test successful, output shape: {sample_output.shape}")
                
        except Exception as e:
            print(f"Initialization info printing error: {str(e)}, but does not affect model usage")

    def get_model(self):
        return self.model

    def get_score_net_dtype(self):
        return next(self.clip_projector.parameters()).dtype

    def predict_weights(self, combined_features):
        """
        Predict weights using MLP network, completely avoiding Transformer structure
        
        Args:
            combined_features: Combined CLIP features and Scores features
                              Shape: [batch_size, 1539]
        Returns:
            Predicted weights
        """
        # Separate features
        clip_features = combined_features[:, :1536]  # CLIP features
        score_features = combined_features[:, 1536:]  # Scores features
        
        # Feature projection
        clip_proj = self.clip_projector(clip_features)    # [batch_size, hidden_size]
        scores_proj = self.scores_projector(score_features)  # [batch_size, hidden_size]
        
        # Concatenate features for fusion
        combined = torch.cat([clip_proj, scores_proj], dim=1)  # [batch_size, hidden_size*2]
        fused = self.fusion_network(combined)  # [batch_size, hidden_size]
        
        # Feature enhancement
        enhanced = self.feature_enhancer(fused) + fused  # Residual connection
        
        # Classification
        weights = self.classifier(enhanced)  # [batch_size, 1]
        
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
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_coido", LlavaConfig_CoIDO)
AutoModelForCausalLM.register(LlavaConfig_CoIDO, LlavaLlamaForCausalLM_CoIDO_ClipScoresFallback) 