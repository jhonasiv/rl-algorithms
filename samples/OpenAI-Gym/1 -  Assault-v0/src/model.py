import functools
import operator
from typing import Iterable

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from rlalgs.value_based.model import DQNModel

patch_typeguard()


class AlienDQN(DQNModel):
    def __init__(self, convolutional_layers: nn.Sequential, linear_layers: list,
                 input_dim: Iterable, device: torch.device):
        super().__init__(convolutional_layers, device)
        self._input_dim = input_dim
        reshape_layer = self.get_last_conv_size()
        first_linear_layer = [n for n, l in enumerate(linear_layers) if isinstance(l, nn.Linear)][0]
        self._linear = nn.Sequential(
            nn.Linear(reshape_layer, linear_layers[first_linear_layer].in_features),
            *linear_layers).to(device)
    
    @typechecked
    def forward(self, x: TensorType[..., "batch"]) -> TensorType[..., "batch"]:
        x = (x.movedim(-1, 0) / 255.).to(self.device)
        x = self._model(x)
        x = x.view(x.size(0), -1)
        x = self._linear(x)
        return x.T
    
    def get_last_conv_size(self):
        num_features_before_fcnn = functools.reduce(operator.mul, list(
                self._model.cpu()(torch.rand(1, *self._input_dim[::-1])).shape))
        self._model = self._model.to(self.device)
        return num_features_before_fcnn
