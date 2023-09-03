from typing import List

from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.gpt2 import GPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp_modules.decoders.base_classes import CaptioningDecoder


class PrefixTuningCaptioner(CaptioningDecoder):
    def __init__(self, z_size, k_prefix=10):
        super().__init__()
        self.gpt: GPT2Model = GPT2Model().to(DEVICE)
        self.prefix_mapper = PrefixMapper(
            prefix_length=k_prefix,
            num_layers=8,
            dim_input=z_size,
            dim_output=self.gpt.embedding_size,
        ).to(DEVICE)

    def forward(self, z_behavior: Tensor, instruction_labels: List[str]) -> CausalLMOutputWithCrossAttentions:
        prefix = self.prefix_mapper(z_behavior)
        output = self.gpt.forward(prefix, instruction_labels)
        return output
