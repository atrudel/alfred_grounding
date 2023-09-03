import torch
from torch import nn, Tensor

from grounding.models.base_models.clip import CLIP_EMBEDDING_SIZE
from grounding.models.base_models.gpt2 import GPT2_EMBEDDING_SIZE


class MultiheadAttention(nn.MultiheadAttention):
    """Adaptation of pytorch's MultiheadAttention that takes care of tranforming the embeddings in Q, K, V
    before feeding them to MHA"""
    def __init__(self, dim_embedding: int, num_heads: int, dropout: float):
        super().__init__(
            embed_dim=dim_embedding,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            batch_first=True
        )
        self.num_heads = num_heads
        self.to_queries = nn.Linear(dim_embedding, dim_embedding)
        self.to_keys_values = nn.Linear(dim_embedding, dim_embedding * 2)

    def forward(self, x: Tensor):
        batch_dim, seq_dim, embed_dim = x.shape
        queries = self.to_queries(x)
        keys_values = self.to_keys_values(x).reshape(batch_dim, seq_dim, 2, embed_dim)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        output, attention_weights = super().forward(queries, keys, values)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, dim_embedding: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.mha = MultiheadAttention(dim_embedding, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim_embedding)
        self.norm2 = nn.LayerNorm(dim_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding, 4 * dim_embedding),
            nn.ReLU(),
            nn.Linear(4 * dim_embedding, dim_embedding)
        )

    def forward(self, x):
        x = x + self.mha(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 dim_embedding: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0.0
                 ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim_embedding, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PrefixMapper(nn.Module):
    def __init__(self,
                 prefix_length: int,
                 num_layers: int = 8,
                 dim_input: int = CLIP_EMBEDDING_SIZE,
                 dim_output: int = GPT2_EMBEDDING_SIZE):
        super().__init__()
        self.prefix_length: int = prefix_length
        self.dim_output: int = dim_output
        self.transformer = Transformer(dim_embedding=dim_output, num_heads=8, num_layers=num_layers)
        self.linear = nn.Linear(dim_input, prefix_length * dim_output)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_output), requires_grad=True)

    def forward(self, z):
        z = self.linear(z).view(z.shape[0], self.prefix_length, self.dim_output)
        constant = self.prefix_const.unsqueeze(0).expand(z.shape[0], *self.prefix_const.shape)
        primer = torch.cat((z, constant), dim=1)
        out = self.transformer(primer)[:, self.prefix_length:]
        return out

# class PrefixMapper(nn.Module):
#     def __init__(self, input_size, gpt_embed_size, k_prefix=10):
#         super().__init__()
#         self.gpt_embed_size: int = gpt_embed_size
#         self.k_prefix: int = k_prefix
#         self.mlp = nn.Sequential(
#             nn.Linear(input_size, input_size * 2),
#             nn.ReLU(),
#             nn.Linear(input_size * 2, k_prefix * gpt_embed_size // 2),
#             nn.ReLU(),
#             nn.Linear(k_prefix * gpt_embed_size // 2, k_prefix * gpt_embed_size)
#         )
#
#     def forward(self, z):
#         out = self.mlp(z)
#         return out.reshape(-1, self.k_prefix, self.gpt_embed_size)
