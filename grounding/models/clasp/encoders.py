from torch import nn
from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection, AutoTokenizer
clip_model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


class BehaviorEncoder(nn.Module):
    def __init__(self, z_size, d_model=512):
        super().__init__()
        self.clip_text_encoder = ...
        self.clip_image_encoder = ...
        self.variational_encoder = VariationalEncoder(
            input_size=d_model,
            hidden_size=d_model // 2,
            z_size=z_size
        )

    def forward(self, images, actions):
        pass


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

    def forward(self, x):
        x = self.mlp(x)
        means = self.fc_means(x)
        log_vars = self.fc_logvars(x)
        return means, log_vars


class TextEncoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_output_size: int = self.clip_text_encoder.config.projection_dim
        self.variational_encoder = VariationalEncoder(
            input_size=clip_output_size,
            hidden_size=clip_output_size // 2,
            z_size=z_size
        )

    def forward(self, text):
        clip_repr = self.clip_text_encoder(text)
        means, log_vars = self.variational_encoder(clip_repr)


if __name__ == '__main__':
    text_encoder = TextEncoder()
