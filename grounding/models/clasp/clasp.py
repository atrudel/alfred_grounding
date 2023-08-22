from typing import List

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput

from config import DEVICE
from grounding.models.clasp.decoders.base_classes import BehaviorGeneratingDecoder, CaptioningDecoder
from grounding.models.clasp.decoders.prefix_tuning.prefix_behavior_generator import PrefixMappingBehaviorGenerator
from grounding.models.clasp.decoders.prefix_tuning.prefix_tuning_captioner import PrefixTuningCaptioner
from grounding.models.clasp.encoders.behavior_encoders import BehaviorEncoder
from grounding.models.clasp.encoders.instruction_encoders import TextEncoder


class CLASP(L.LightningModule):
    def __init__(self, z_size: int, beta_align: float = 1, beta_caption: float = 1, beta_behavior_gen: float = 1,
                 temperature: float = 0.07, learning_rate: float = 1e-4, weightdecay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.instruction_encoder: TextEncoder = TextEncoder(z_size=z_size)
        self.behavior_encoder: BehaviorEncoder = BehaviorEncoder(z_size=z_size)
        self.captioner: CaptioningDecoder = PrefixTuningCaptioner(z_size=z_size)
        self.behavior_generator: BehaviorGeneratingDecoder = PrefixMappingBehaviorGenerator(z_size)
        self.cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.beta_align: float = beta_align
        self.beta_caption: float = beta_caption
        self.beta_behavior_gen: float = beta_behavior_gen
        self.temperature: float = temperature
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weightdecay

    def training_step(self, batch, batch_idx) -> float:
        instructions, images, actions = batch
        loss = self._forward(actions, images, instructions)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        instructions, images, actions = batch
        loss = self._forward(actions, images, instructions)
        self.log("val_loss", loss, batch_size=len(instructions))


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=self.weight_decay,
                                 amsgrad=False)

    def predict_step(self, batch, batch_idx):
        pass

    def _forward(self, actions: List[str], images: List[object], instructions: List[str]) -> float:
        loss_align: float = self._forward_align(instructions, images, actions)
        loss_caption: float = self._forward_captioning(instructions, images, actions)
        loss_behavior_gen: float = self._forward_behavior_generation(instructions, images, actions)
        loss_global: float = self.beta_align * loss_align + \
                             self.beta_caption * loss_caption + \
                             self.beta_behavior_gen * loss_behavior_gen
        return loss_global

    def _forward_align(self, instructions: Tensor, images: Tensor, actions: Tensor) -> float:
        z_instruction = self.reparametrization_trick(*self.instruction_encoder(instructions))
        z_behavior = self.reparametrization_trick(*self.behavior_encoder(images, actions))
        loss_align = self.contrastive_loss(z_instruction, z_behavior)
        return loss_align

    def _forward_captioning(self, instructions, images, actions) -> float:
        z_behavior = self.reparametrization_trick(*self.behavior_encoder(images, actions))
        output: CausalLMOutputWithCrossAttentions = self.captioner(z_behavior, instructions)
        return output.loss.mean()

    def _forward_behavior_generation(self, instructions, images, actions) -> float:
        z_instruction = self.reparametrization_trick(*self.instruction_encoder(instructions))
        output: ModelOutput = self.behavior_generator(z_instruction, images, actions)
        return output.loss.mean()

    def contrastive_loss(self, z_text, z_behavior):
        batch_size: int = z_text.shape[0]
        z_text = F.normalize(z_text, dim=1)
        z_behavior = F.normalize(z_behavior, dim=1)

        similarity_logits = torch.matmul(z_text, z_behavior.T)

        labels = torch.arange(batch_size).to(DEVICE)
        loss_text = self.cross_entropy(similarity_logits, labels)
        loss_behav = self.cross_entropy(similarity_logits.T, labels)
        loss = (loss_text + loss_behav) / 2
        return loss


    def reparametrization_trick(self, means, log_vars):
        # Todo: log_vars or vars
        stds = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(stds)
        z = eps * stds + means
        return z
