from typing import Tuple

import torch
from torch import nn, Tensor

from grounding.models.frozen_models.clip import CLIPModel


class BehaviorEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.clip = CLIPModel
        self.variational_encoder = VariationalEncoder(
            input_size=self.clip.text_embedding_dim() + self.clip.image_embedding_dim(),
            hidden_size=(self.clip.text_embedding_dim() + self.clip.image_embedding_dim()) // 2,
            z_size=z_size
        )

    def forward(self, images, actions) -> Tuple[Tensor, Tensor]:
        images_clip_repr: Tensor = self.clip.encode_images(images)  # [B, 512]
        actions_clip_repr: Tensor = self.clip.encode_texts(actions)  # [B, 512]
        behavior_repr: Tensor = torch.cat([images_clip_repr, actions_clip_repr], dim=1)  # [B, 1024]

        means, log_vars = self.variational_encoder(behavior_repr)
        return means, log_vars



class TextEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.clip = CLIPModel
        self.variational_encoder = VariationalEncoder(
            input_size=self.clip.text_embedding_dim(),
            hidden_size=self.clip.text_embedding_dim() // 2,
            z_size=z_size
        )

    def forward(self, text) -> Tuple[Tensor, Tensor]:
        clip_repr = self.clip.encode_texts(text)
        means, log_vars = self.variational_encoder(clip_repr)
        return means, log_vars


class VariationalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, z_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.fc_means = nn.Linear(hidden_size, z_size)
        self.fc_logvars = nn.Linear(hidden_size, z_size)

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        x = self.mlp(x)
        means = self.fc_means(x)
        log_vars = self.fc_logvars(x)
        return means, log_vars
