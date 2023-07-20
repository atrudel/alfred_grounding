import torch
from torch import nn
from torch.nn import MultiheadAttention


class Captioner(nn.Module):
    def __init__(self, embed_dim, k_prefix = 10, n_layers = 8):
        super().__init__()
        self.prefix_mapper = PrefixMapper(embed_dim, k_prefix, n_layers)
        self.gpt = ... # Todo: GPT2
        self.cross_entropy = nn.CrossEntropyLoss()

    def train_forward(self, z, instructions):
        prefix = self.prefix_mapper(z)
        instructions_tok = ...
        gpt_input = torch.cat([prefix, instructions_tok], dim=1)
        output = self.gpt(gpt_input)
        # todo: get loss


class PrefixMapper(nn.Module):
    def __init__(self, embed_dim, k_prefix = 10, n_layers = 8):
        super().__init__()
        self.k_prefix = k_prefix
        self.mapper = nn.Sequential(nn.ModuleList(
            [nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
            )
            ] * n_layers)
        )

    def forward(self, z):
        # Todo [chiant] find a Huggingface module that implements generation
        ...