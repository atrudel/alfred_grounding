import abc
from typing import List

from torch import nn, Tensor
from transformers.utils import ModelOutput


class BehaviorGeneratingDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, z, images, command_labels):
        raise NotImplementedError


class CaptioningDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, z: Tensor, instruction_labels: List[str]) -> ModelOutput:
        raise NotImplementedError
