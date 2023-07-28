import abc

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from grounding.models.frozen_models.clip import CLIPModel
from grounding.models.frozen_models.gpt2 import PrefixGPT2Model


class TextDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, z, labels):
        raise NotImplementedError


class BehaviorDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, z, images, labels):
        raise NotImplementedError


class PrefixMappingTextDecoder(TextDecoder):
    def __init__(self, embed_dim, k_prefix=10, n_layers=8):
        super().__init__()
        self.prefix_mapper = PrefixMapper(embed_dim, k_prefix, n_layers)
        self.prefix_gpt: PrefixGPT2Model = PrefixGPT2Model

    def forward(self, z, labels) -> CausalLMOutputWithCrossAttentions:
        prefix = self.prefix_mapper(z)
        output = self.prefix_gpt.forward(prefix, labels)
        return output


class PrefixMappingBehaviorDecoder(BehaviorDecoder):
    def __init__(self, z_dim, k_prefix=10, n_layers=8):
        super().__init__()
        self.clip = CLIPModel
        self.prefix_mapper: PrefixMapper = PrefixMapper(z_dim + self.clip.image_embedding_dim(), k_prefix, n_layers)
        self.prefix_gpt = PrefixGPT2Model

    def forward(self, z, images, labels) -> CausalLMOutputWithCrossAttentions:
        image_repr = self.clip.encode_images(images)
        joint_repr = torch.cat([z, image_repr], dim=1)
        prefix = self.prefix_mapper(joint_repr)
        output = self.prefix_gpt.forward(prefix, labels)
        return output


class PrefixMapper(nn.Module):
    def __init__(self, z_dim, gpt_dim, k_prefix=10):
        super().__init__()
        self.gpt_dim: int = gpt_dim
        self.k_prefix: int = k_prefix
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, gpt_dim * k_prefix // 2),
            nn.ReLU(),
            nn.Linear(gpt_dim * k_prefix // 2, gpt_dim * k_prefix)
        )

    def forward(self, z):
        out = self.mlp(z)
        return out.reshape(-1, self.k_prefix, self.gpt_dim)



# class PrefixMapper(nn.Module):
#     def __init__(self, z_dim, gpt_dim, k_prefix=10, n_layers=8):
#         super().__init__()
#         self.k_prefix = k_prefix
#         self.mapper = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=z_dim, num_heads=8)
#             for _ in range(n_layers)
#         ])
#         self.fc = nn.Linear(z_dim, gpt_dim)
#
#     def forward(self, z):
#         # Todo [chiant] find a Huggingface module that implements generation
#         for attention_layer in self.layers:
#             z, _ = attention_layer(z, z, z) # todo: verifier
#         prefix = self.fc(z)
#         return prefix

    