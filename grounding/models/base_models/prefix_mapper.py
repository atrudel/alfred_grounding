from torch import nn


class PrefixMapper(nn.Module):
    def __init__(self, input_size, gpt_embed_size, k_prefix=10):
        super().__init__()
        self.gpt_embed_size: int = gpt_embed_size
        self.k_prefix: int = k_prefix
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, k_prefix * gpt_embed_size // 2),
            nn.ReLU(),
            nn.Linear(k_prefix * gpt_embed_size // 2, k_prefix * gpt_embed_size)
        )

    def forward(self, z):
        out = self.mlp(z)
        return out.reshape(-1, self.k_prefix, self.gpt_embed_size)
