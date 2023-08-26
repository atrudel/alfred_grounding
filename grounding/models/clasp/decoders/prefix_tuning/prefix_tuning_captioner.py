from typing import List

from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.gpt2 import GPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp.decoders.base_classes import CaptioningDecoder


class PrefixTuningCaptioner(CaptioningDecoder):
    def __init__(self, z_size, k_prefix=10):
        super().__init__()
        self.gpt: GPT2Model = GPT2Model().to(DEVICE)
        self.prefix_mapper = PrefixMapper(
            input_size=z_size,
            gpt_embed_size=self.gpt.embedding_size,
            k_prefix=k_prefix
        ).to(DEVICE)

    def forward(self, z: Tensor, instruction_labels: List[str]) -> CausalLMOutputWithCrossAttentions:
        prefix = self.prefix_mapper(z)
        output = self.gpt.forward(prefix, instruction_labels)
        return output
