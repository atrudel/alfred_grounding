from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from grounding.models.base_models.gpt2 import PrefixGPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp.decoders.base_classes import CaptionDecoder


class PrefixTuningCaptioner(CaptionDecoder):
    def __init__(self, embed_dim, k_prefix=10, n_layers=8):
        super().__init__()
        self.prefix_mapper = PrefixMapper(embed_dim, k_prefix, n_layers)
        self.prefix_gpt: PrefixGPT2Model = PrefixGPT2Model

    def forward(self, z, labels) -> CausalLMOutputWithCrossAttentions:
        prefix = self.prefix_mapper(z)
        output = self.prefix_gpt.forward(prefix, labels)
        return output
