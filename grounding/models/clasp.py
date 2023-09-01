from typing import List, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput

from config import DEVICE
from grounding.data_processing.action import Action
from grounding.models.clasp_modules.decoders.base_classes import BehaviorGeneratingDecoder, CaptioningDecoder
from grounding.models.clasp_modules.decoders.prefix_tuning.prefix_behavior_generator import PrefixTuningBehaviorGenerator
from grounding.models.clasp_modules.decoders.prefix_tuning.prefix_tuning_captioner import PrefixTuningCaptioner
from grounding.models.clasp_modules.encoders.behavior_encoders import BehaviorEncoder
from grounding.models.clasp_modules.encoders.instruction_encoders import TextEncoder


class CLASP(L.LightningModule):
    def __init__(self, z_size: int, beta_align: float = 1, beta_caption: float = 1, beta_behavior_gen: float = 1,
                 temperature: float = 0.07, learning_rate: float = 1e-4, weightdecay: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.instruction_encoder: TextEncoder = TextEncoder(z_size=z_size)
        self.behavior_encoder: BehaviorEncoder = BehaviorEncoder(z_size=z_size)
        self.captioner: CaptioningDecoder = PrefixTuningCaptioner(z_size=z_size)
        self.behavior_generator: BehaviorGeneratingDecoder = PrefixTuningBehaviorGenerator(z_size)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weightdecay

    def training_step(self, batch, batch_idx) -> Tensor:
        loss: Tensor = self._forward(batch)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        loss: Tensor = self._forward(batch)
        self.log("val_loss", loss.item())
        # alignment_accuarcy = self._alignment_accuracy(batch)
        # self.log("val_acc_alignment", alignment_accuarcy)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=self.weight_decay,
                                 amsgrad=False)

    def predict_step(self, batch, batch_idx):
        pass

    def _forward(self, batch: dict) -> Tensor:
        loss_align: Tensor = self._forward_alignment(batch)
        loss_caption: Tensor = self._forward_captioning(batch)
        loss_behavior_gen: Tensor = self._forward_behavior_generation(batch)
        loss_global: Tensor = self.hparams.beta_align * loss_align + \
                             self.hparams.beta_caption * loss_caption + \
                             self.hparams.beta_behavior_gen * loss_behavior_gen
        return loss_global

    def _forward_alignment(self, batch) -> Tensor:
        z_instruction: Tensor = self._encode_instructions(batch["instruction_clip_feats"])
        z_behavior: Tensor = self._encode_behaviors(batch["image_clip_feats"], batch["command_clip_feats"])
        loss_align = self.contrastive_loss(z_instruction, z_behavior)
        return loss_align

    def _forward_captioning(self, batch) -> Tensor:
        z_behavior: Tensor = self._encode_behaviors(batch["image_clip_feats"], batch["command_clip_feats"])
        output: CausalLMOutputWithCrossAttentions = self.captioner(
            z_behavior, batch["instruction"]
        )
        return output.loss.mean()

    def _forward_behavior_generation(self, batch) -> Tensor:
        z_instruction: Tensor = self._encode_instructions(batch["instruction_clip_feats"])
        output: ModelOutput = self.behavior_generator(
            z_instruction,
            batch["image_clip_feats"],
            batch["command"]
        )
        return output.loss.mean()

    def _encode_instructions(self, instruction_clip_feats: Tensor) -> Tensor:
        z_instruction: Tensor = self.reparametrization_trick(*self.instruction_encoder(
            instruction_clip_feats
        ))
        return z_instruction

    def _encode_behaviors(self, image_clip_feats: Tensor, command_clip_feats: Tensor) -> Tensor:
        z_behavior: Tensor = self.reparametrization_trick(*self.behavior_encoder(
            image_clip_feats, command_clip_feats
        ))
        return z_behavior

    def contrastive_loss(self, z_text, z_behavior) -> Tensor:
        batch_size: int = z_text.shape[0]
        z_text = F.normalize(z_text, dim=1, p=2)
        z_behavior = F.normalize(z_behavior, dim=1, p=2)

        similarity_matrix = torch.matmul(z_text, z_behavior.T)
        logits = similarity_matrix / self.hparams.temperature

        labels = torch.arange(batch_size).to(DEVICE)

        loss_text = self.cross_entropy(logits, labels)
        loss_behav = self.cross_entropy(logits.T, labels)
        loss = (loss_text + loss_behav) / 2
        return loss

    def reparametrization_trick(self, means, log_vars) -> Tensor:
        stds = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(stds)
        z = eps * stds + means
        return z

    def evaluate_command_generation_on_all_object_options(self,
                                                          action: Action,
                                                          candidate_output_texts: List[str]
                                                          ) -> Tuple[Tensor, Tensor]:
        """
        The model scores all candidate commands in order to figure out which one is preferred.
        :param action: Action object associated with the action being tested
        :param candidate_output_texts: Command options will all permutations of the objec of interaction
        :return: logits Tensor
        """
        n_candidates = len(candidate_output_texts)
        z_instruction = self.reparametrization_trick(*self.instruction_encoder(
            action.instruction_clip_features
        ))
        batched_z: Tensor = z_instruction.repeat(n_candidates, 1) # [n, 512]
        batched_image_feats: Tensor = action.image_clip_features.repeat(n_candidates, 1) # [n, 512]
        with torch.no_grad():
            output: ModelOutput = self.behavior_generator(
                batched_z,
                batched_image_feats,
                candidate_output_texts
            )
        logits: Tensor = output.logits

        ouput_tokenized: BatchEncoding = self.behavior_generator.gpt.tokenizer(
            candidate_output_texts, return_tensors='pt',
            padding="max_length", max_length=logits.shape[1]
        )
        output_toks: Tensor = ouput_tokenized["input_ids"]
        return logits, output_toks

    def _alignment_accuracy(self, batch) -> float:
        pass






