from typing import List

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.gpt2 import GPT2Model
from grounding.models.base_models.prefix_mapper import AttentionPrefixMapper, MLPPrefixMapper
from grounding.models.clasp_modules.decoders.base_classes import BehaviorGeneratingDecoder
from grounding.models.base_models.clip import CLIP_EMBEDDING_SIZE


class PrefixTuningBehaviorGenerator(BehaviorGeneratingDecoder):
    def __init__(self, z_size, k_prefix=10, attention=True):
        super().__init__()
        self.gpt = GPT2Model()
        if attention:
            self.prefix_mapper: AttentionPrefixMapper = AttentionPrefixMapper(
                prefix_length=k_prefix,
                num_layers=8,
                dim_input=z_size + CLIP_EMBEDDING_SIZE,
                dim_output=self.gpt.embedding_size,
            ).to(DEVICE)
        else:
            self.prefix_mapper = MLPPrefixMapper(
                input_size=z_size + CLIP_EMBEDDING_SIZE,
                gpt_embed_size=self.gpt.embedding_size,
                k_prefix=k_prefix
            )

    def forward(self,
                z: Tensor,
                images_clip_encoded: Tensor,
                command_labels: List[str]) -> CausalLMOutputWithCrossAttentions:
        prefix: Tensor = self._make_prefix(z, images_clip_encoded)
        output: CausalLMOutputWithCrossAttentions = self.gpt.forward(prefix, command_labels)
        return output

    def generate(self, z: Tensor, images_clip_encoded: Tensor) -> List[str]:
        prefixes: Tensor = self._make_prefix(z, images_clip_encoded)
        commands: List[str] = self.gpt.generate_with_prefix_embeddings(prefixes)
        return commands

    def _make_prefix(self, z, images_clip_encoded):
        joint_repr: Tensor = torch.cat([z, images_clip_encoded.reshape(-1, CLIP_EMBEDDING_SIZE)], dim=1)
        prefix: Tensor = self.prefix_mapper(joint_repr)
        return prefix

