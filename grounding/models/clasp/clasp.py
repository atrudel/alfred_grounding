import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import lightning as L

from grounding.data_processing.datasets import get_train_and_val_dataloaders
from grounding.models.clasp.decoders.prefix_tuning.prefix_tuning_captioner import PrefixTuningCaptioner
from grounding.models.clasp.decoders.t5.t5_captioner import T5Captioner
from grounding.models.clasp.decoders.prefix_tuning.prefix_behavior_generator import PrefixMappingBehaviorGenerator
from grounding.models.clasp.decoders.t5.t5_behavior_generator import T5BehaviorGenerator
from grounding.models.clasp.decoders.base_classes import BehaviorDecoder, TextDecoder
from grounding.models.clasp.encoders.instruction_encoders import TextEncoder
from grounding.models.clasp.encoders.behavior_encoders import BehaviorEncoder

device = "gpu" if torch.cuda.is_available() else "cpu"


class CLASP(L.LightningModule):
    def __init__(self, z_size: int,
                 prefix_mapping: bool = False,
                 beta_align: float = 1,
                 beta_caption: float = 1,
                 beta_behavior_gen: float = 1,
                 temperature: float = 0.07,
                 learning_rate: float = 1e-4,
                 weight_deacay: float = 0.01):
        super().__init__()
        self.instruction_encoder: TextEncoder = TextEncoder(z_size=z_size)
        self.behavior_encoder: BehaviorEncoder = BehaviorEncoder(z_size=z_size)
        self.captioner: TextDecoder = PrefixTuningCaptioner(embed_dim=z_size) if prefix_mapping \
            else T5Captioner(z_size=z_size)
        self.behavior_generator: BehaviorDecoder = PrefixMappingBehaviorGenerator(z_size) if prefix_mapping \
            else T5BehaviorGenerator(z_size=z_size)
        self.cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.beta_align: float = beta_align
        self.beta_caption: float = beta_caption
        self.beta_behavior_gen: float = beta_behavior_gen
        self.temperature: float = temperature
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_deacay

    def training_step(self, batch, batch_idx) -> float:
        instructions, images, actions = batch
        loss = self._forward(actions, images, instructions)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        instructions, images, actions = batch
        loss = self._forward(actions, images, instructions)
        self.log("val_loss", loss)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=self.weight_decay,
                                 amsgrad=False)

    def _forward(self, actions, images, instructions) -> float:
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
        return output.loss

    def _forward_behavior_generation(self, instructions, images, actions) -> float:
        z_instruction = self.reparametrization_trick(*self.instruction_encoder(instructions))
        output = self.behavior_generator(z_instruction, images, actions)
        return output.loss


    def contrastive_loss(self, z_text, z_behavior):
        batch_size: int = z_text.shape[0]
        z_text = F.normalize(z_text, dim=1)
        z_behavior = F.normalize(z_behavior, dim=1)

        similarity_logits = torch.matmul(z_text, z_behavior.T)

        labels = torch.arange(batch_size)
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


if __name__ == '__main__':
    z_size = 512
    clasp = CLASP(z_size=z_size, prefix_mapping=True)

    train_dataloader, val_dataloader = get_train_and_val_dataloaders(batch_size=8, use_raw_images=True, num_workers=2)
    trainer = L.Trainer(limit_train_batches=2, max_epochs=1)
    trainer.fit(model=clasp, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # z_size = 4
    # clasp = CLASP(z_size=z_size, temperature=1, captioner=PrefixMappingTextDecoder(embed_dim=z_size))
    # z_text = Tensor([[1,2,3,4],
    #                  [0,0,0,0],
    #                  [4,4,4,4]])
    # z_behavior = Tensor([[1,2,3,4],
    #                  [1,1,1,1],
    #                  [4,4,4,4]])
    # print(clasp.contrastive_loss(z_text, z_behavior))
    # print(clasp.contrastive_loss_complex(z_text, z_behavior))