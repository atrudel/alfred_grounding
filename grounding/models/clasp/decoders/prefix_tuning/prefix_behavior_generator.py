import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.gpt2 import GPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp.decoders.base_classes import BehaviorGeneratingDecoder
from grounding.models.base_models.clip import CLIP_EMBEDDING_SIZE


class PrefixTuningBehaviorGenerator(BehaviorGeneratingDecoder):
    def __init__(self, z_dim, k_prefix=10):
        super().__init__()
        self.gpt = GPT2Model()
        self.prefix_mapper: PrefixMapper = PrefixMapper(
            input_size=z_dim + CLIP_EMBEDDING_SIZE,
            gpt_embed_size=self.gpt.embedding_size,
            k_prefix=k_prefix
        ).to(DEVICE)

    def forward(self, z, images_clip_encoded, command_labels) -> CausalLMOutputWithCrossAttentions:
        joint_repr = torch.cat([z, images_clip_encoded.squeeze()], dim=1)
        prefix = self.prefix_mapper(joint_repr)
        output = self.gpt.forward(prefix, command_labels)
        return output
