from __future__ import annotations

from copy import deepcopy

import torch
from abc import ABC, abstractmethod
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


class BaseDQNModel(nn.Module, ABC):
    def copy(self) -> BaseDQNModel:
        """
        Generates a deep copy of this instance
        
        :return: new object
        """
        return deepcopy(self)
    
    @abstractmethod
    def forward(self, x):
        """
        Process the input data
        
        :param x: input data
        :return: output data
        """


class DQNModel(BaseDQNModel, ABC):
    def __init__(self, sequential_model: nn.Sequential, seed: int = 0, ):
        """
        DQNModel constructor.
        
        :param sequential_model: the sequential model for this neural network, including
        activation functions
        :param seed: random seed
        """
        super(DQNModel, self).__init__()
        torch.manual_seed(seed)
        self._model = sequential_model
    
    @typechecked
    def forward(self, x: TensorType["batch", -1]) -> TensorType["batch", -1, -1]:
        for layer in self._model[:-1]:
            x = layer(x)
        x = self._model[-1](x)
        x = x.refine_names("action", "value")
        return x


class DuelingDQNModel(BaseDQNModel, ABC):
    def __init__(self, hidden_sequential: nn.Sequential, value_sequential: nn.Sequential,
                 advantage_sequential: nn.Sequential, seed: int, device: torch.device,
                 last_layer_operation: str = "mean"):
        """
        DuelingDQNModel constructor.
        
        :param input_features: number of input features
        :param output_features: number of output features
        :param seed: random seed
        :param device: torch device
        :param last_layer_operation: operation to use to aggregate the value and advantage functions
        """
        super().__init__()
        operations = {'max': torch.max, 'mean': torch.mean}
        self._operation = operations.get(last_layer_operation, "mean")
        torch.manual_seed(seed)
        self._hidden_layers = hidden_sequential.to(device)
        self._value_layers = value_sequential.to(device)
        self._advantage_layers = advantage_sequential.to(device)
    
    @typechecked
    def forward(self, x: TensorType["batch", -1]) -> TensorType["batch", -1, -1]:
        x = self._hidden_layers(x)
        
        value = self._value_layers(x)
        advantage = self._advantage_layers(x)
        
        q = value + advantage - self._operation(advantage, dim=1, keepdim=True)
        return q
