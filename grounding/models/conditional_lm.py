from typing import Optional, Tuple, List

import torch
from torch import nn, Tensor
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast, BatchEncoding
from transformers.modeling_outputs import Seq2SeqLMOutput


class ImageConditionedLLMOnDecoder(nn.Module):
    def __init__(self, concat_dim=1024, hidden_dim=512, use_image=True):
        super(ImageConditionedLLMOnDecoder, self).__init__()
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
        self.tokenizer: T5Tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        self.concat_dim: int = concat_dim
        self.hidden_dim: int = hidden_dim
        self.use_image: bool = use_image
        self.modality_fusion_module: Optional[nn.Module] = nn.Linear(self.concat_dim, self.hidden_dim) if use_image else None

    def train_forward(self,
                      input_token_ids: Tensor,
                      input_att_mask: Tensor,
                      decoder_input_token_ids: Tensor,
                      decoder_input_att_mask: Tensor,
                      image_features: Tensor,
                      output_token_ids: Tensor
                      ):
        decoder_input_embeddings = self.model.decoder.embed_tokens(decoder_input_token_ids)
        if self.use_image:
            fused_decoder_input_embeddings = self.modality_fusion_module(
                torch.cat([image_features, decoder_input_embeddings], dim=2)
            )
        else:
            fused_decoder_input_embeddings = decoder_input_embeddings

        return self.model(
            input_ids=input_token_ids,
            attention_mask=input_att_mask,
            decoder_inputs_embeds=fused_decoder_input_embeddings,
            decoder_attention_mask=decoder_input_att_mask,
            labels=output_token_ids
        )

    def test_generate(self,
                      input_token_ids: Tensor,
                      decoder_input_token_ids: Tensor,
                      decoder_input_att_mask: Tensor,
                      image_feature_sequence: Tensor,
                      stop_token: int,
                      ):

        decoder_input_embeddings = self.model.decoder.embed_tokens(decoder_input_token_ids[:,:stop_token])
        if self.use_image:
            image_feature_sequence = image_feature_sequence[:,:stop_token]
            fused_decoder_input_embeddings = self.modality_fusion_module(
                torch.cat([image_feature_sequence, decoder_input_embeddings], dim=2)
            )
        else:
            fused_decoder_input_embeddings = decoder_input_embeddings

        decoder_input_att_mask = decoder_input_att_mask[:,:stop_token]

        output: Seq2SeqLMOutput = self.model(
            input_ids=input_token_ids,
            decoder_inputs_embeds=fused_decoder_input_embeddings,
            decoder_attention_mask=decoder_input_att_mask,
        )

        return output


    def prepare_image_and_output_data(self, input_image_feats: Tensor, output_texts: List[str]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pad_tok: int = self.tokenizer.pad_token_id
        ignore_tok: int = -100

        output_tokenized: BatchEncoding = self.tokenizer(output_texts, padding='longest', return_tensors='pt')

        # Preparing output tokens for label
        output_toks = output_tokenized['input_ids'].clone()
        output_toks[output_toks == pad_tok] = ignore_tok

        # Preparing input tokens and attention mask for the decoder
        decoder_input_toks = output_tokenized['input_ids'].roll(shifts=1, dims=1)
        decoder_input_toks[:,0] = pad_tok
        decoder_input_att_mask = (decoder_input_toks != pad_tok).float()
        decoder_input_att_mask[:,0] = 1.  # Start of sentence token is the pad token so it should be attended to


        # Preparing image features tensor: copy to match the token sequence length
        sequence_length: int = decoder_input_toks.shape[1]
        decoder_image_features = input_image_feats.repeat(1, sequence_length, 1)

        return decoder_input_toks, decoder_input_att_mask, decoder_image_features, output_toks



    @classmethod
    def load(cls, save_path: str):
        checkpoint = torch.load(save_path)
        model_state_dict = checkpoint['model_state_dict']
        model = ImageConditionedLLMOnDecoder()
        model.load_state_dict(model_state_dict)
        model.eval()
        return model
