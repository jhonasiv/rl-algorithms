from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from rlalgs.value_based.model import DQNModel

patch_typeguard()


class BasicModel(DQNModel):
    def __init__(self, sequential_model: nn.Sequential, seed: int = 0):
        super().__init__(sequential_model, seed)
    
    @typechecked
    def forward(self, x: TensorType["batch", -1]) -> TensorType["batch", -1, -1]:
        x = self._model(x)
        x = x.unsqueeze(1)
        return x
