from typing import Tuple

from torch import nn, Tensor

from config import DEVICE
from grounding.models.clasp.encoders.variational_encoder import VariationalEncoder
from models.base_models.clip import CLIP_EMBEDDING_SIZE


class TextEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.variational_encoder = VariationalEncoder(
            input_size=CLIP_EMBEDDING_SIZE,
            hidden_size=CLIP_EMBEDDING_SIZE // 2,
            z_size=z_size
        ).to(DEVICE)

    def forward(self, text_clip_encoded) -> Tuple[Tensor, Tensor]:
        means, log_vars = self.variational_encoder(text_clip_encoded.squeeze()) # [B, 512]
        return means, log_vars
