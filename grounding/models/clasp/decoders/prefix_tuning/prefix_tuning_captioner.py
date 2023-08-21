from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from grounding.models.base_models.gpt2 import PrefixGPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp.decoders.base_classes import CaptioningDecoder


class PrefixTuningCaptioner(CaptioningDecoder):
    def __init__(self, z_size, k_prefix=10):
        super().__init__()
        self.prefixable_gpt: PrefixGPT2Model = PrefixGPT2Model()
        self.prefix_mapper = PrefixMapper(
            input_size=z_size,
            gpt_embed_size=self.prefixable_gpt.embedding_size,
            k_prefix=k_prefix
        )

    def forward(self, z, labels) -> CausalLMOutputWithCrossAttentions:
        prefix = self.prefix_mapper(z)
        output = self.prefixable_gpt.forward(prefix, labels)
        return output
