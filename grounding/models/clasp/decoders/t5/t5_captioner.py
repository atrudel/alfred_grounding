from typing import List

import torch
from torch import Tensor
from transformers.modeling_outputs import Seq2SeqLMOutput

from grounding.models.clasp.decoders.base_classes import CaptionDecoder
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder


class T5Captioner(CaptionDecoder):
    def __init__(self, z_size: int):
        super().__init__()
        self.conditional_t5: ImageConditionedLLMOnDecoder = ImageConditionedLLMOnDecoder(z_size)

    def forward(self, z: Tensor, labels: List[str]) -> Seq2SeqLMOutput:
        text_teacher_forcing_ids, \
            decoder_input_att_mask, \
            decoder_input_z, \
            output_toks = self.conditional_t5.prepare_decoder_input_output_data(z, labels)
        text_teacher_forcing_embeds: Tensor = self.conditional_t5.model.decoder.embed_tokens(text_teacher_forcing_ids)
        decoder_input_embeds = self.conditional_t5.modality_fusion_module(
            torch.cat([decoder_input_z, text_teacher_forcing_embeds], dim=2)
        )
        result: Seq2SeqLMOutput = self.conditional_t5.model.decoder.forward(
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_input_att_mask,
            labels=output_toks ## todo labels doensn't exist as an argument.
        )
        return result
