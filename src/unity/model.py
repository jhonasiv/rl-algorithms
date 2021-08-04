import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from value_based.model import BaseDQNModel, DuelingDQNModel

patch_typeguard()


class BallModel(BaseDQNModel):
    def __init__(self, sequential_model: nn.Sequential, rotation_x_layer: nn.Sequential,
                 rotation_z_layer: nn.Sequential, device: torch.device, seed: int = 0):
        """
        DQNModel constructor.

        :param sequential_model: the sequential model for this neural network, including
        activation functions
        :param seed: random seed
        """
        super().__init__()
        torch.manual_seed(seed)
        self.device = device
        self._model = sequential_model.to(self.device)
        self._rotation_z_layer = rotation_z_layer.to(self.device)
        self._rotation_x_layer = rotation_x_layer.to(self.device)
    
    @typechecked
    def forward(self, input: TensorType["batch", -1]) -> TensorType["batch", 2, -1]:
        x = self._model(input)
        rotation_z = self._rotation_z_layer(x).unsqueeze(1)
        rotation_x = self._rotation_x_layer(x).unsqueeze(1)
        x = torch.cat((rotation_x, rotation_z), dim=1)
        return x


class DuelingBallModel(DuelingDQNModel):
    def __init__(self, hidden_sequential: nn.Sequential, value_sequential: nn.Sequential,
                 advantage_sequential: nn.Sequential, rotation_x_layer: nn.Sequential,
                 rotation_z_layer: nn.Sequential, device: torch.device, seed: int,
                 last_layer_operation: str = "mean"):
        super().__init__(hidden_sequential=hidden_sequential, value_sequential=value_sequential,
                         advantage_sequential=advantage_sequential, seed=seed, device=device,
                         last_layer_operation=last_layer_operation)
        self._rotation_x_layer = rotation_x_layer.to(device)
        self._rotation_z_layer = rotation_z_layer.to(device)
    
    @typechecked
    def forward(self, x: TensorType["batch", -1]) -> TensorType["batch", 2, -1]:
        x = self._hidden_layers(x)
        value = self._value_layers(x)
        adv = self._advantage_layers(x)
        q = value + adv - self._operation(adv, dim=1, keepdim=True)
        x_rotation = self._rotation_x_layer(q).unsqueeze(1)
        z_rotation = self._rotation_z_layer(q).unsqueeze(1)
        x = torch.cat((x_rotation, z_rotation), dim=1)
        return x
