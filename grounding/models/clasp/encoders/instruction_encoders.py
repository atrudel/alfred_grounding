from typing import Tuple

import torch
from torch import nn, Tensor

from grounding.models.base_models.clip import CLIPModelFrozen
from grounding.models.clasp.encoders.variational_encoder import VariationalEncoder


class TextEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.clip = CLIPModelFrozen()
        self.variational_encoder = VariationalEncoder(
            input_size=self.clip.text_embedding_dim(),
            hidden_size=self.clip.text_embedding_dim() // 2,
            z_size=z_size
        )

    def forward(self, text) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            clip_repr = self.clip.encode_texts(text)
        means, log_vars = self.variational_encoder(clip_repr)
        return means, log_vars
