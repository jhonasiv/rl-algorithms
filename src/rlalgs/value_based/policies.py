from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from rlalgs.utils.functions import casted_exponential_function, constant_decay_function
from rlalgs.value_based.annealing import LinearAnnealing

patch_typeguard()


class GaussianNoise:
    """
    Gaussian noise for continuous action spaces
    """
    
    @typechecked
    def __init__(self, noise_percentile, action_bounds: TensorType['action', "boundary"],
                 evaluation_noise: TensorType["action", "noise", torch.float32], ):
        """
        Gaussian Noise constructor.
        
        :param action_bounds: agent action space boundaries with shape (num_actions, *),
        where the first row is the minimum bounds of all actions and the second is the maximum
        :param noise_percentile: gaussian noise percentiles for each action in the action space
        :param evaluation_noise: noise to be used when evaluating
        """
        self.noise_percentile = noise_percentile
        self.evaluation_noise = evaluation_noise
        self.action_bounds = action_bounds
    
    @typechecked
    def step(self, action_values: TensorType["action", "value"]):
        action_values = action_values.squeeze()
        noise = torch.normal(action_values, std=torch.abs(self.noise_percentile * action_values))
        values = (action_values + noise).clamp(min=self.action_bounds[0], max=self.action_bounds[1])
        return values


class BasePolicy(ABC):
    @abstractmethod
    def step(self, time_step: int) -> float:
        """
        Calculate epsilon for this time step

        :param time_step: time step in episode
        :return: updated epsilon
        """
    
    @abstractmethod
    @typechecked
    def exploit(self, action_values: TensorType["action", "value"]) -> TensorType["action"]:
        """
        Takes an exploitative action

        :param action_values: set of action values
        :return: a 1D Tensor representing the chosen action(s)
        """
    
    @abstractmethod
    @typechecked
    def explore(self, action_values: TensorType["action", "value"]) -> TensorType["action"]:
        """
        Takes an exploratory action

        :param action_values: set of action values
        :return: a 1D Tensor representing the chosen action(s)
        """


class BaseEpsilonGreedyPolicy(BasePolicy, ABC):
    def __init__(self, epsilon: float, discrete: bool,
                 action_boundaries: Optional[TensorType] = None,
                 noise: Optional[GaussianNoise] = None, seed: int = 0):
        
        torch.random.manual_seed(seed)
        self._epsilon = epsilon
        self._action_boundaries = action_boundaries
        self._discrete = discrete
        self._noise = noise
        assert self._discrete or self._noise is not None, (
                "If policy is not discrete, a gaussian " "noise must be defined.")
    
    @typechecked
    def exploit(self, action_values: TensorType["action": ..., "batch"]) -> np.ndarray:
        if self._discrete:
            actions = torch.argmax(action_values, dim=0)
            return actions.detach().cpu().numpy()
        else:
            return self._noise.step(action_values).detach().cpu().numpy()
    
    @typechecked
    def explore(self, action_values: TensorType["action": ..., "batch"]) -> np.ndarray:
        if self._discrete:
            result = torch.randint(0, action_values.shape[0], size=action_values.shape[1:])
            return result.detach().cpu().numpy()
        else:
            return ((self._action_boundaries[:, 1] - self._action_boundaries[:, 0]) * torch.rand(
                    len(action_values)) + self._action_boundaries[:, 0]).detach().cpu().numpy()


class ConstantEpsilonGreedy(BaseEpsilonGreedyPolicy):
    def __init__(self, epsilon: float, discrete: bool,
                 action_boundaries: Optional[TensorType["action", "boundary"]] = None,
                 noise: Optional[GaussianNoise] = None):
        super().__init__(epsilon, discrete, action_boundaries, noise)
    
    @typechecked
    def step(self, time_step: int) -> float:
        return self._epsilon


@dataclass
class DecayEpsilonGreedy(BaseEpsilonGreedyPolicy):
    
    def __init__(self, epsilon: float, discrete: bool, epsilon_min: float,
                 epsilon_decay_rate: float,
                 action_boundaries: Optional[TensorType["action", "boundary"]] = None,
                 noise: Optional[GaussianNoise] = None, seed: int = 0):
        """
        Epsilon greedy policy where

         epsilon = epsilon * epsilon_decay_rate^(time step)
         
        :param epsilon: initial value for epsilon
        :param discrete: if the action space is discrete
        :param epsilon_min: minimum value for epsilon
        :param epsilon_decay_rate: rate that the epsilon decays after each time step
        :param action_boundaries: boundary values for the actions, only for continuous action spaces
        :param noise: noise to be added to the action values, only for continuous action spaces
        """
        super().__init__(epsilon, discrete, action_boundaries, noise)
        self._epsilon_min = epsilon_min
        self._epsilon_decay_rate = epsilon_decay_rate
    
    @typechecked
    def step(self, *args) -> float:
        self._epsilon = constant_decay_function(self._epsilon, self._epsilon_decay_rate)
        self._epsilon = max(self._epsilon, self._epsilon_min)
        return self._epsilon


@dataclass
class ExponentialEpsilonGreedy(BaseEpsilonGreedyPolicy):
    
    def __init__(self, epsilon: float, discrete: bool, epsilon_min: float, exp_k: float,
                 exp_b: float, action_boundaries: Optional[TensorType["action", "boundary"]],
                 noise: Optional[GaussianNoise], seed: int = 0):
        """
        Exponentially calculate epsilon, following this equation
        
        e^(exp_k * int((time_step / exp_b) ** 2)
        
        :param epsilon: initial value for epsilon
        :param discrete: if the action space is discrete
        :param epsilon_min: minimum value for epsilon
        :param exp_k: multiplying term in the equation
        :param exp_b: term that the time step is divided by
        :param action_boundaries: boundary values for the actions, only for continuous action spaces
        :param noise: noise to be added to the action values, only for continuous action spaces
        """
        super().__init__(epsilon, discrete, action_boundaries, noise)
        self._epsilon_min = epsilon_min
        self._exp_k = exp_k
        self._exp_b = exp_b
    
    @typechecked
    def step(self, time_step: int) -> float:
        self._epsilon = casted_exponential_function(k=self._exp_k, x=time_step, b=self._exp_b,
                                                    exp=2, a=1)
        self._epsilon = max(self._epsilon, self._epsilon_min)
        return self._epsilon


class LinearAnnealedEpsilon(BaseEpsilonGreedyPolicy):
    def __init__(self, epsilon: float, discrete: bool, epsilon_min: float, num_steps: int):
        super().__init__(epsilon, discrete)
        self.func = LinearAnnealing(val_init=epsilon, val_thresh=epsilon_min, max_itt=num_steps)
    
    def step(self, time_step: int) -> float:
        self._epsilon = self.func.step(time_step)
        return self._epsilon
