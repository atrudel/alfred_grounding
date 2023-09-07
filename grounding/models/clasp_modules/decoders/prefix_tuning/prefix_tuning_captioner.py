from typing import List

from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.gpt2 import GPT2Model
from grounding.models.base_models.prefix_mapper import AttentionPrefixMapper, MLPPrefixMapper
from grounding.models.clasp_modules.decoders.base_classes import CaptioningDecoder


class PrefixTuningCaptioner(CaptioningDecoder):
    def __init__(self, z_size, k_prefix=10, attention=True):
        super().__init__()
        self.gpt: GPT2Model = GPT2Model().to(DEVICE)
        if attention:
            self.prefix_mapper = AttentionPrefixMapper(
                prefix_length=k_prefix,
                num_layers=8,
                dim_input=z_size,
                dim_output=self.gpt.embedding_size,
            ).to(DEVICE)
        else:
            self.prefix_mapper = MLPPrefixMapper(
                input_size=z_size,
                gpt_embed_size=self.gpt.embedding_size,
                k_prefix=k_prefix
            )

    def forward(self, z_behavior: Tensor, instruction_labels: List[str]) -> CausalLMOutputWithCrossAttentions:
        prefix = self.prefix_mapper(z_behavior)
        output = self.gpt.forward(prefix, instruction_labels)
        return output
