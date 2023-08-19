from torch import nn


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


# Version avec des self-attention layers
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