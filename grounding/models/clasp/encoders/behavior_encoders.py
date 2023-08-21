from typing import Tuple

import torch
from torch import nn, Tensor

from config import DEVICE
from grounding.models.base_models.clip import CLIPModelFrozen
from grounding.models.clasp.encoders.variational_encoder import VariationalEncoder


class BehaviorEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.clip = CLIPModelFrozen()
        self.variational_encoder = VariationalEncoder(
            input_size=self.clip.text_embedding_dim() + self.clip.image_embedding_dim(),
            hidden_size=(self.clip.text_embedding_dim() + self.clip.image_embedding_dim()) // 2,
            z_size=z_size
        ).to(DEVICE)

    def forward(self, images, actions) -> Tuple[Tensor, Tensor]:
        images_clip_repr: Tensor = self.clip.encode_images(images)  # [B, 512]
        actions_clip_repr: Tensor = self.clip.encode_texts(actions)  # [B, 512]
        behavior_repr: Tensor = torch.cat([images_clip_repr, actions_clip_repr], dim=1)  # [B, 1024]

        means, log_vars = self.variational_encoder(behavior_repr)
        return means, log_vars
