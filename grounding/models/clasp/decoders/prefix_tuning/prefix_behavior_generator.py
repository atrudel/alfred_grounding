import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.models.base_models.clip import CLIPModelFrozen
from grounding.models.base_models.gpt2 import PrefixGPT2Model
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.models.clasp.decoders.base_classes import BehaviorGeneratingDecoder


class PrefixMappingBehaviorGenerator(BehaviorGeneratingDecoder):
    def __init__(self, z_dim, k_prefix=10):
        super().__init__()
        self.clip = CLIPModelFrozen()
        self.prefix_gpt = PrefixGPT2Model()
        self.prefix_mapper: PrefixMapper = PrefixMapper(
            input_size=z_dim + self.clip.image_embedding_dim(),
            gpt_embed_size=self.prefix_gpt.embedding_size,
            k_prefix=k_prefix
        ).to(DEVICE)

    def forward(self, z, images, labels) -> CausalLMOutputWithCrossAttentions:
        image_repr = self.clip.encode_images(images)
        joint_repr = torch.cat([z, image_repr], dim=1)
        prefix = self.prefix_mapper(joint_repr)
        output = self.prefix_gpt.forward(prefix, labels)
        return output
