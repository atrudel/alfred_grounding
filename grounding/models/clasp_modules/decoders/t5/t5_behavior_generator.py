import torch
from torch import Tensor
from transformers.modeling_outputs import Seq2SeqLMOutput

from grounding.models.clasp_modules.decoders.base_classes import BehaviorGeneratingDecoder
from grounding.models.conditional_lm import ImageConditionedLLMOnDecoder


class T5BehaviorGenerator(BehaviorGeneratingDecoder):
    def __init__(self ,z_size: int):
        super().__init__()
        self.conditional_t5: ImageConditionedLLMOnDecoder = ImageConditionedLLMOnDecoder(z_size)

    def forward(self, z, images, labels):
        text_teacher_forcing_ids, \
            decoder_input_att_mask, \
            decoder_input_images, \
            output_toks = self.conditional_t5.prepare_decoder_input_output_data(images, labels)
        text_teacher_forcing_embeds: Tensor = self.conditional_t5.model.decoder.embed_tokens(text_teacher_forcing_ids)
        decoder_input_embeds = self.conditional_t5.modality_fusion_module(
            torch.cat([decoder_input_images, text_teacher_forcing_embeds], dim=2)
        )
        result: Seq2SeqLMOutput = self.conditional_t5.model.decoder.forward(
            encoder_hidden_states=z,
            inputs_embeds=decoder_input_embeds,
            attention_mask=decoder_input_att_mask,
            labels=output_toks
        )
        return result