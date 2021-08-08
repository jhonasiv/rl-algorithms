import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Collection

patch_typeguard()


class Grid:
    def __init__(self, lower_bounds: Collection[float], upper_bounds: Collection[float],
                 n_tilings: Collection[int]):
        """
        Create a grid for a number of elements equal to the size of the boundaries.
        
        :param lower_bounds: list of lower boundaries for each element being tiled.
        :param upper_bounds: list of upper boundaries for each element being tiled.
        :param n_tilings: number of tilings for each element being tiled.
        """
        self._tilings = torch.stack([torch.linspace(start=lb, end=up, steps=nt) for lb, up, nt in
                                     zip(lower_bounds, upper_bounds, n_tilings)])
        self._step_size = torch.from_numpy(
                (np.array(upper_bounds) - np.array(lower_bounds)) / n_tilings)
    
    @typechecked
    def batch_to_continuous(self, discrete_values: TensorType["batch", Any]) -> TensorType[
        "batch", -1]:
        discrete_values = discrete_values.long().T
        value = self._tilings.to(discrete_values.device).gather(1, discrete_values).T
        value += torch.rand(value.shape[-1], device=value.device) * self._step_size.to(value.device)
        return value
    
    @typechecked
    def batch_to_discrete(self, continuous_values: TensorType["batch", Any]) -> TensorType[
        "batch", -1]:
        mapping = map(lambda x, y: torch.bucketize(x, y), continuous_values.T, self._tilings)
        value = torch.stack(list(mapping)).T
        return value
