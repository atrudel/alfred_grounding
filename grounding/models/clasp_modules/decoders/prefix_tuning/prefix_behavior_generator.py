from typing import List

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.gpt2 import GPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp_modules.decoders.base_classes import BehaviorGeneratingDecoder
from grounding.models.base_models.clip import CLIP_EMBEDDING_SIZE


class PrefixTuningBehaviorGenerator(BehaviorGeneratingDecoder):
    def __init__(self, z_size, k_prefix=10):
        super().__init__()
        self.gpt = GPT2Model()
        self.prefix_mapper: PrefixMapper = PrefixMapper(
            prefix_length=k_prefix,
            num_layers=8,
            dim_input=z_size + CLIP_EMBEDDING_SIZE,
            dim_output=self.gpt.embedding_size,
        ).to(DEVICE)

    def forward(self,
                z: Tensor,
                images_clip_encoded: Tensor,
                command_labels: List[str]) -> CausalLMOutputWithCrossAttentions:
        joint_repr: Tensor = torch.cat([z, images_clip_encoded.squeeze()], dim=1)
        prefix: Tensor = self.prefix_mapper(joint_repr)
        output: CausalLMOutputWithCrossAttentions = self.gpt.forward(prefix, command_labels)
        return output
