from typing import List

import lightning as L
import torch
from torch import Tensor, nn
from transformers import BatchEncoding, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from config import DEVICE
from grounding.data_processing.datasets_train import get_train_and_val_dataloaders
from grounding.models.base_models.prefix_mapper import PrefixMapper
from grounding.training.utils import get_grad_norm


class StandalonePrefixTuningGPT2(L.LightningModule):
    def __init__(self, prefix_length: int, learning_rate: float = 1e-4, weightdecay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()

        # Frozen GPT-2 model
        self.gpt: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        for param in self.gpt.parameters():
            param.requires_grad = False

        # Tokenizer
        self.tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefix mapping network
        self.prefix_mapper: PrefixMapper = PrefixMapper(
            input_size=512,  # Resnet feature size
            gpt_embed_size=self.gpt.config.n_embd,  # 768
            k_prefix=self.hparams.prefix_length,
        ).to(DEVICE)

    def training_step(self, batch, batch_idx) -> float:
        instructions, image_resnet_features, commands = batch
        loss, perplexity = self(
            list(instructions),
            image_resnet_features,
            list(commands)
        )
        batch_size = len(instructions)
        grad_norm = get_grad_norm(self)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_perplexity", perplexity, batch_size=batch_size)
        self.log("train_grad_norm", grad_norm, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        instructions, image_resnet_features, commands = batch
        loss, perplexity = self(
            list(instructions),
            image_resnet_features,
            list(commands)
        )
        batch_size = len(instructions)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_perplexity", perplexity, batch_size=batch_size)

    def forward(self,
                instructions: List[str],
                resnet_features: Tensor,
                commands: List[str]):

        # Tokenize
        instruction_tokenized: BatchEncoding = self.tokenizer(instructions, return_tensors='pt', padding=True)
        commands_tokenized: BatchEncoding = self.tokenizer(commands, return_tensors='pt', padding=True)

        # Create the prompt
        instruction_embeds: Tensor = self.embed_tokens(instruction_tokenized.input_ids)  # [B, len_i, 768]
        prefix_embeds: Tensor = self.prefix_mapper(resnet_features)  # [B, k, 768]
        command_embeds: Tensor = self.embed_tokens(commands_tokenized.input_ids)  # [B, len_c, 768]
        prompt_embeds: Tensor = torch.cat([    # [B, len_i + k + len_c, 768]
            instruction_embeds,
            prefix_embeds,
            command_embeds
        ], dim=1)

        # Create the attention mask
        attention_mask: Tensor = torch.cat([  # [B, len_i + k + len_c]
            instruction_tokenized.attention_mask,
            torch.ones(prefix_embeds.shape[:2]),
            commands_tokenized.attention_mask
        ], dim=1)

        # Create label - Ignore the prompt
        labels = torch.cat([    # [B, len_i + k + len_c]
            torch.full(size=instruction_embeds.shape[:2], fill_value=-100),
            torch.full(size=prefix_embeds.shape[:2], fill_value=-100),
            commands_tokenized.input_ids
        ], dim=1)
        output: CausalLMOutputWithCrossAttentions = self.gpt(
            inputs_embeds=prompt_embeds.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
            labels=labels.to(DEVICE),
            return_dict=True
        )
        loss = output.loss
        perplexity = torch.exp(loss)
        return loss, perplexity

    def configure_optimizers(self):
        return torch.optim.AdamW(self.prefix_mapper.parameters(),
                                 lr=self.hparams.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=self.hparams.weightdecay,
                                 amsgrad=False)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        embeddings: nn.Module = self.gpt.get_input_embeddings()
        return embeddings(tokens.to(DEVICE))





if __name__ == '__main__':
    train_dataloader, val_dataloader = get_train_and_val_dataloaders(8, clasp_mode=False)
    prefix_gpt = StandalonePrefixTuningGPT2(5)
    for batch in train_dataloader:
        prefix_gpt.training_step(batch, 1)

