from torch import nn, Tensor
import torch
import torch.nn.functional as F

from grounding.models.clasp.captioning import Captioner
from grounding.models.clasp.encoders import TextEncoder, BehaviorEncoder

device = "gpu" if torch.cuda.is_available() else "cpu"


class CLASP(nn.Module):
    def __init__(self, z_size, temperature: float = 0.07):
        super().__init__()
        self.text_encoder: TextEncoder = TextEncoder(z_size=z_size)
        self.behavior_encoder: BehaviorEncoder = BehaviorEncoder(z_size=z_size)
        self.captioner: Captioner()
        self.temperature: float = temperature

    def train_align(self, instructions: Tensor, images: Tensor, actions: Tensor):
        z_text = self.reparametrization_trick(
            *self.text_encoder(instructions)
        )
        z_behavior = self.reparametrization_trick(
            *self.behavior_encoder(images, actions)
        )
        loss_align = self.contrastive_loss(z_text, z_behavior)
        return loss_align

    def train_captioning(self, instructions, images, actions):
        ...

    def train_behavior_generation(self):
        ...

    def contrastive_loss(self, z_text, z_behavior):
        batch_size: int = z_text.shape[0]
        z_text = F.normalize(z_text, dim=1)
        z_behavior = F.normalize(z_behavior, dim=1)

        similarity_logits = torch.matmul(z_text, z_behavior.T)

        labels = torch.arange(batch_size)
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        loss_text = cross_entropy(similarity_logits, labels)
        loss_behav = cross_entropy(similarity_logits.T, labels)
        loss = (loss_text + loss_behav) / 2
        return loss


    def reparametrization_trick(self, means, log_vars):
        # Todo: log_vars or vars
        stds = torch.exp(0.5 * log_vars)
        eps = torch.randn_like(stds)
        z = eps * stds + means
        return z


if __name__ == '__main__':
    clasp = CLASP(z_size=4, temperature=1)
    z_text = Tensor([[1,2,3,4],
                     [0,0,0,0],
                     [4,4,4,4]])
    z_behavior = Tensor([[1,2,3,4],
                     [1,1,1,1],
                     [4,4,4,4]])
    print(clasp.contrastive_loss(z_text, z_behavior))
    print(clasp.contrastive_loss_complex(z_text, z_behavior))