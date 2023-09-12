from typing import List

import torch
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BatchEncoding, T5Tokenizer
from transformers.generation import BeamSearchEncoderDecoderOutput

from config import DEVICE
from grounding.models.clasp_modules.decoders.base_classes import BehaviorGeneratingDecoder
from grounding.models.base_models.clip import CLIP_EMBEDDING_SIZE


class T5BehaviorGenerator(BehaviorGeneratingDecoder):
    def __init__(self ,z_size: int):
        super().__init__()
        self.t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True).to(DEVICE)
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.fc_image: nn.Module = nn.Linear(CLIP_EMBEDDING_SIZE, self.t5.config.d_model).to(DEVICE)
        self.fc_z: nn.Module = nn.Linear(z_size, self.t5.config.d_model).to(DEVICE)
        self.z_size = z_size

    def forward(self, z: Tensor,
                images_clip_encoded: Tensor,
                command_labels: List[str]):
        prompt: Tensor = self._propagate_prompt(z, images_clip_encoded)
        labels: Tensor = self._prepare_labels(command_labels)  # [B, L]
        output = self.t5(
            inputs_embeds=prompt.to(DEVICE),
            labels=labels.to(DEVICE)
        )
        return output

    def generate(self, z_instruction: Tensor, image_clip_features: Tensor):
        prompt: Tensor = self._propagate_prompt(z_instruction, image_clip_features)
        outputs: BeamSearchEncoderDecoderOutput = self.t5.generate(
            inputs_embeds=prompt,
            num_beams=5,
            do_sample=False
        )
        output_sequences: List[str] = self.tokenizer.batch_decode(outputs)
        return output_sequences


    def _propagate_prompt(self, z, images_clip_encoded) -> Tensor:
        z_prompt: Tensor = self.fc_z(z.reshape(-1, 1, self.z_size))  # [B, 1, 512]
        image_prompt: Tensor = self.fc_image(images_clip_encoded.reshape(-1, 1, CLIP_EMBEDDING_SIZE))  # [B, 1, 512]
        prompt: Tensor = torch.cat([z_prompt, image_prompt], dim=1)  # [B, 2, 512]
        return prompt

    def _prepare_labels(self, command_labels: List[str]) -> Tensor:
        tokenized: BatchEncoding = self.tokenizer(command_labels, return_tensors='pt', padding=True)
        labels: Tensor = tokenized.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels



    # def forward(self, z, images, labels):
    #     text_teacher_forcing_ids, \
    #         decoder_input_att_mask, \
    #         decoder_input_images, \
    #         output_toks = self.conditional_t5.prepare_decoder_input_output_data(images, labels)
    #     text_teacher_forcing_embeds: Tensor = self.conditional_t5.model.decoder.embed_tokens(text_teacher_forcing_ids)
    #     decoder_input_embeds = self.conditional_t5.modality_fusion_module(
    #         torch.cat([decoder_input_images, text_teacher_forcing_embeds], dim=2)
    #     )
    #     result: Seq2SeqLMOutput = self.conditional_t5.model.decoder.forward(
    #         encoder_hidden_states=z,
    #         inputs_embeds=decoder_input_embeds,
    #         attention_mask=decoder_input_att_mask,
    #         labels=output_toks
    #     )
    #     return result
