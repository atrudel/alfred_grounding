from torch import nn


class BehaviorGenerator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.text_encoder =