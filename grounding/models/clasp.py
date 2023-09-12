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
from grounding.models.base_models.clip import CLIPModelFrozen
from grounding.models.clasp_modules.decoders.t5.t5_behavior_generator import T5BehaviorGenerator


class CLASP(L.LightningModule):
    def __init__(self, z_size: int, beta_align: float = 1, beta_caption: float = 1, beta_behavior_gen: float = 1,
                 temperature: float = 0.07, learning_rate: float = 1e-4, weightdecay: float = 0.01,
                 attention_prefix_tuning=True, alignment_only=False):
        super().__init__()
        self.save_hyperparameters()
        self.instruction_encoder: TextEncoder = TextEncoder(z_size=z_size)
        self.behavior_encoder: BehaviorEncoder = BehaviorEncoder(z_size=z_size)
        if not alignment_only:
            self.captioner: CaptioningDecoder = PrefixTuningCaptioner(z_size=z_size, attention=attention_prefix_tuning)
            # self.behavior_generator: BehaviorGeneratingDecoder = PrefixTuningBehaviorGenerator(z_size, attention=attention_prefix_tuning)
            self.behavior_generator: BehaviorGeneratingDecoder = T5BehaviorGenerator(z_size=z_size)

        self.cross_entropy = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx) -> Tensor:
        self.train()
        loss: Tensor = self._forward(batch)
        self.log("train_loss", loss.item(), batch_size=len(batch['instruction']))
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self.eval()
        with torch.no_grad():
            loss: Tensor = self._forward(batch)
            self.log("val_loss", loss.item(), batch_size=len(batch['instruction']))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=self.hparams.weightdecay,
                                 amsgrad=False)

    def _forward(self, batch: dict) -> Tensor:
        loss_align: Tensor = self._forward_alignment(batch)

        if self.hparams.alignment_only:
            loss_global: Tensor = loss_align
        else:
            # loss_caption: Tensor = self._forward_captioning(batch)
            loss_behavior_gen: Tensor = self._forward_behavior_generation(batch)
            # loss_global: Tensor = self.hparams.beta_align * loss_align + \
            #                      self.hparams.beta_behavior_gen * loss_behavior_gen
                                 # self.hparams.beta_caption * loss_caption + \
        # return loss_global
        return loss_behavior_gen

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
            z=z_instruction,
            images_clip_encoded=batch["image_clip_feats"],
            command_labels=batch["command"]
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

    def generate_behavior(self, instruction_clip_features, image_clip_features: Tensor) -> str:
        z_instruction = self._encode_instructions(instruction_clip_features).reshape(1, -1)  # [B, 512]
        predicted = self.behavior_generator.generate(z_instruction, image_clip_features)
        return predicted

    def evaluate_candidate_commands_on_all_object_options(self,
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

    def evaluate_candidate_commands_through_alignment(self,
                                                      action: Action,
                                                      candidate_output_texts: List[str]):
        # Encode instruction
        z_instruction = self._encode_instructions(action.instruction_clip_features)  # [512]

        # Extract CLIP embeddings for all candidate commands
        clip: CLIPModelFrozen = CLIPModelFrozen()
        command_embeds: Tensor = clip.encode_texts(candidate_output_texts)  # [80, 512]

        # Encode all candidate commands with the image as different behaviors
        zs_behavior: Tensor = self._encode_behaviors(  # [80, 512]
            image_clip_feats=action.image_clip_features.repeat(command_embeds.shape[0], 1),
            command_clip_feats=command_embeds
        )
        # Compute similarity of each behavior with the instruction
        similarities: Tensor = zs_behavior @ z_instruction  # [80]
        return similarities


