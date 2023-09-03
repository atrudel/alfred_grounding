from typing import List

import torch
from torch import Tensor, nn
from transformers import BatchEncoding, GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE

GPT2_EMBEDDING_SIZE = 768


class GPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, prefix_embeddings: Tensor, target_texts: List[str]) -> CausalLMOutputWithCrossAttentions:
        batch_size, prefix_length, embed_size = prefix_embeddings.shape

        # Get embeddings associated with target text
        target_tokenized: BatchEncoding = self.tokenizer(target_texts, return_tensors='pt', padding=True)
        target_embeddings: Tensor = self.embed_tokens(target_tokenized.input_ids)

        # Concatenate prefix and target embeddings
        input_embeddings: Tensor = torch.cat([prefix_embeddings, target_embeddings], dim=1)
        input_attention_mask: Tensor = torch.cat([
            torch.ones(size=(batch_size, prefix_length)),  # Pay attention to the prefix
            target_tokenized.attention_mask
        ], dim=1).to(DEVICE)

        # Prepare labels (ignoring the prefix)
        labels: Tensor = torch.cat([
            torch.full(size=prefix_embeddings.shape[:2], fill_value=-100),  # Ignore the prefix to calculate the loss
            target_tokenized.input_ids
        ], dim=1).to(DEVICE)
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore pad tokens in loss calculation

        output: CausalLMOutputWithCrossAttentions = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=input_attention_mask,
            labels=labels,
            return_dict=True
        )
        return output

    def generate(self, prompts: List[str], max_length: int = 50, do_sample: bool = False):
        output_texts = []
        for prompt in prompts:
            inputs: BatchEncoding = self.tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                output = self.model.generate(
                    inputs=inputs.input_ids.to(DEVICE),
                    attention_mask=inputs.attention_mask.to(DEVICE),
                    max_length=max_length,
                    do_sample=do_sample,
                    num_beams=10
                )
            decoded_output: List[str] = self.tokenizer.batch_decode(output)
            generated_text: str = decoded_output[0][len(prompt):]
            output_texts.append(generated_text)
        return output_texts

    @property
    def embedding_size(self):
        return self.model.config.n_embd

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        embeddings: nn.Module = self.model.get_input_embeddings()
        return embeddings(tokens.to(DEVICE))

