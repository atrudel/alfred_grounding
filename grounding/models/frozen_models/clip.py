import warnings
from typing import List

from torch import Tensor
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizerFast, \
    CLIPImageProcessor


class CLIPModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip_checkpoint = "openai/clip-vit-base-patch32"
        text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_checkpoint)
        tokenizer = CLIPTokenizerFast.from_pretrained(clip_checkpoint)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_checkpoint)
        image_processor = CLIPImageProcessor.from_pretrained(clip_checkpoint)

    @classmethod
    def encode_texts(cls, text: List[str], padding: bool = True) -> Tensor:
        """Tokenizes and encodes a batch of strings and returns the batched embeddings"""
        inputs = cls.tokenizer(text, padding=padding, return_tensors="pt")
        outputs = cls.text_encoder(**inputs)
        return outputs.text_embeds

    @classmethod
    def encode_images(cls, images: List[Tensor]) -> Tensor:
        """Processes and encodes a batch of images and returns the batched embeddings"""
        inputs = cls.image_processor(images, return_tensors='pt')
        outputs = cls.image_encoder(**inputs)
        return outputs.image_embeds

    @classmethod
    def text_embedding_dim(cls) -> int:
        return cls.text_encoder.config.projection_dim

    @classmethod
    def image_embedding_dim(cls) -> int:
        return cls.image_encoder.config.projection_dim
