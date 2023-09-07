from typing import List, Union

import numpy as np
from torch import Tensor
from torch import nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizerFast, \
    CLIPImageProcessor
from transformers import logging

from config import DEVICE
from typing import List, Union

import numpy as np
from torch import Tensor
from torch import nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizerFast, \
    CLIPImageProcessor
from transformers import logging

from config import DEVICE

CLIP_EMBEDDING_SIZE = 512



class CLIPModelFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        clip_checkpoint = "openai/clip-vit-base-patch32"

        logging.set_verbosity_error()  # Silence Huggingface's warnings

        # CLIP Text Encoder
        self.tokenizer: CLIPTokenizerFast = CLIPTokenizerFast.from_pretrained(clip_checkpoint)
        self.text_encoder: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(clip_checkpoint).to(DEVICE)

        # CLIP Text Decoder
        self.image_processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(clip_checkpoint)
        self.image_encoder: CLIPVisionModelWithProjection = CLIPVisionModelWithProjection.from_pretrained(clip_checkpoint).to(DEVICE)

        logging.set_verbosity_warning()


        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def encode_texts(self, text: Union[str, List[str]], padding: bool = True) -> Tensor:
        """Tokenizes and encodes a batch of strings and returns the batched embeddings"""
        inputs = self.tokenizer(text, padding=padding, return_tensors="pt")
        outputs = self.text_encoder(
            input_ids=inputs['input_ids'].to(DEVICE),
            attention_mask=inputs['attention_mask'].to(DEVICE)
        )
        return outputs.text_embeds

    def encode_images(self, images: Union[np.ndarray, List[Tensor]]) -> Tensor:
        """Processes and encodes a batch of images and returns the batched embeddings"""
        inputs = self.image_processor(images, return_tensors='pt')
        outputs = self.image_encoder(
            pixel_values=inputs['pixel_values'].to(DEVICE)
        )
        return outputs.image_embeds

    def text_embedding_dim(self) -> int:
        return self.text_encoder.config.projection_dim

    def image_embedding_dim(self) -> int:
        return self.image_encoder.config.projection_dim


if __name__ == '__main__':
    clip = CLIPModelFrozen()