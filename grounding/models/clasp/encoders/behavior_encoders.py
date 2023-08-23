from typing import Tuple

import torch
from torch import nn, Tensor

from config import DEVICE
from grounding.models.clasp.encoders.variational_encoder import VariationalEncoder
from models.base_models.clip import CLIP_EMBEDDING_SIZE


class BehaviorEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.variational_encoder = VariationalEncoder(
            input_size=CLIP_EMBEDDING_SIZE * 2,
            hidden_size=CLIP_EMBEDDING_SIZE,
            z_size=z_size
        ).to(DEVICE)

    def forward(self, images_clip_encoded, commands_clip_encoded) -> Tuple[Tensor, Tensor]:
        behavior_repr: Tensor = torch.cat(
            [images_clip_encoded.squeeze(), commands_clip_encoded.squeeze()],
            dim=1)  # [B, 1024]
        means, log_vars = self.variational_encoder(behavior_repr)
        return means, log_vars
