import torch
from DrTorch.modules import TrainableModule


class BertFreezed(TrainableModule):
    def __init__(self):
        super().__init__()

    def forward(self, kwards: dict) -> torch.Tensor:
        pass