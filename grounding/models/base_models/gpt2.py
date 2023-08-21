from typing import List

import torch
from torch import Tensor, nn
from transformers import GPT2TokenizerFast, BatchEncoding, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class PrefixGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2")

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, prefix_embeddings: Tensor, target_texts: List[str]) -> CausalLMOutputWithCrossAttentions:

        # Get embeddings associated with target text
        target_tokenized: BatchEncoding = self.tokenizer(target_texts, return_tensors='pt', padding=True)
        target_embeddings: Tensor = self.model.get_input_embeddings()(target_tokenized.input_ids)

        # Concatenate prefix and target embeddings
        input_embeddings: Tensor = torch.cat([prefix_embeddings, target_embeddings], dim=1)
        input_attention_mask: Tensor = torch.cat([
            torch.ones(prefix_embeddings.shape[:2]),
            target_tokenized.attention_mask
        ], dim=1)
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=input_embeddings,
                attention_mask=input_attention_mask,
            )
        return outputs

    def generate(self, prompts: List[str], max_length: int = 50, do_sample: bool = False):
        output_texts = []
        for prompt in prompts:
            inputs: BatchEncoding = self.tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                output = self.model.generate(inputs=inputs.input_ids,
                                            attention_mask=inputs.attention_mask,
                                            max_length=max_length,
                                            do_sample=do_sample)
            decoded_output: List[str] = self.tokenizer.batch_decode(output)
            generated_text: str = decoded_output[0][len(prompt):]
            output_texts.append(generated_text)
        return output_texts


