from abc import ABC, abstractmethod
from typing import Collection, Optional

import torch
from torch.nn import functional as f
from torchtyping import patch_typeguard
from typeguard import typechecked

from rlalgs.value_based.annealing import BaseFunction
from rlalgs.value_based.model import BaseDQNModel
from rlalgs.value_based.policies import BasePolicy
from rlalgs.value_based.replay import (BaseBuffer, BaseUpdateVariant, Experience,
                                       PrioritizedReplayBuffer, ProportionalUpdateVariant,
                                       ReplayBuffer)
from rlalgs.value_based.strategy.estimate import BaseEstimatorStrategy

patch_typeguard()


class BaseLearningStrategy(ABC):
    memory: BaseBuffer
    
    @abstractmethod
    @typechecked
    def learn(self, experiences: Collection[Experience], estimation_strategy: BaseEstimatorStrategy,
              q_target: BaseDQNModel, q_local: BaseDQNModel, policy: BasePolicy,
              optimizer: torch.optim.Optimizer, gamma: float, episode: Optional[int]) -> None:
        """
        Learning strategy
        
        :param policy: policy followed by the agent
        :param episode: current time step
        :param experiences: sample of experiences
        :param estimation_strategy: strategy to estimate local and target action-values
        :param q_target: target network
        :param q_local: local network
        :param optimizer: torch optimizer
        :param gamma: discount rate
        """


class DQNLearningStrategy(BaseLearningStrategy):
    @typechecked
    def __init__(self, batch_size: int, buffer_size: int, seed: int, device: torch.device):
        """
        DQN learning strategy constructor
        
        :param batch_size: replay buffer minibatch size
        :param buffer_size: replay buffer maximum size
        :param seed: random seed
        :param device: exec device
        """
        self.memory = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size, seed=seed,
                                   device=device)
    
    @typechecked
    def learn(self, experiences: Collection[Experience], estimation_strategy: BaseEstimatorStrategy,
              q_target: BaseDQNModel, q_local: BaseDQNModel, policy: BasePolicy,
              optimizer: torch.optim.Optimizer, gamma: float, **kwargs) -> None:
        states, actions, rewards, next_states, dones = experiences
        target_estimate, local_estimate = estimation_strategy.estimate(q_target=q_target,
                                                                       q_local=q_local,
                                                                       policy=policy, states=states,
                                                                       next_states=next_states,
                                                                       rewards=rewards,
                                                                       actions=actions, dones=dones,
                                                                       gamma=gamma)
        optimizer.zero_grad()
        loss = f.mse_loss(local_estimate, target_estimate)
        loss.backward()
        optimizer.step()


class PrioritizedLearningStrategy(BaseLearningStrategy):
    def __init__(self, batch_size: int, buffer_size: int, seed: int, device: torch.device,
                 alpha: BaseFunction, beta: BaseFunction,
                 update_variant: BaseUpdateVariant = ProportionalUpdateVariant(1e-5)):
        """
        Prioritized Learning Strategy constructor
        
        :param batch_size: replay buffer minibatch size
        :param buffer_size: replay buffer maximum size
        :param seed: random seed
        :param device: exec device
        :param alpha: prioritization exponent
        :param beta: importance-sampling bias exponent
        :param update_variant: replay buffer priority update variant
        """
        self.memory: PrioritizedReplayBuffer = PrioritizedReplayBuffer(batch_size=batch_size,
                                                                       buffer_size=buffer_size,
                                                                       seed=seed, device=device,
                                                                       alpha=alpha,
                                                                       update_variant=update_variant)
        self.alpha = alpha
        self.beta = beta
    
    def learn(self, experiences: Collection[Experience], estimation_strategy: BaseEstimatorStrategy,
              q_target: BaseDQNModel, q_local: BaseDQNModel, policy: BasePolicy,
              optimizer: torch.optim.Optimizer, gamma: float, episode: int) -> None:
        beta = self.beta.step(episode)
        self.alpha.step(episode)
        states, actions, rewards, next_states, dones, samples_id = experiences
        
        target_estimate, local_estimate = estimation_strategy.estimate(q_target=q_target,
                                                                       q_local=q_local,
                                                                       policy=policy, states=states,
                                                                       next_states=next_states,
                                                                       rewards=rewards,
                                                                       actions=actions, dones=dones,
                                                                       gamma=gamma)
        weights = self.memory.calc_is_weight(samples_id=samples_id, beta=beta)
        td_error = (target_estimate - local_estimate).squeeze()
        loss = (td_error ** 2 * weights).mean()
        self.memory.update(torch.abs(td_error), samples_id)
        optimizer.zero_grad()
        loss.backward()
        for param in q_local.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
