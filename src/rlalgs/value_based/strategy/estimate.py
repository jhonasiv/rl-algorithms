from abc import ABC, abstractmethod
from typing import Collection, Tuple

import torch
from torchtyping import TensorType
from typeguard import typechecked

from rlalgs.value_based.model import BaseDQNModel
from rlalgs.value_based.policies import BasePolicy


class BaseEstimatorStrategy(ABC):
    @staticmethod
    @abstractmethod
    def estimate(q_target: BaseDQNModel, q_local: BaseDQNModel, policy: BasePolicy,
                 states: Collection, next_states: Collection, rewards: Collection,
                 actions: Collection, dones: Collection, gamma: float):
        """
        Estimate target and local action values
        
        :param policy: policy followed by the agent
        :param q_target: target network
        :param q_local: local network
        :param states: states minibatch
        :param next_states: next states minibatch
        :param rewards: rewards minibatch
        :param actions: actions minibatch
        :param dones: dones minibatch
        :param gamma: discount rate
        :return: target and local estimate
        """


class DQNEstimatorStrategy(BaseEstimatorStrategy):
    @staticmethod
    @typechecked
    def estimate(q_target: BaseDQNModel, q_local: BaseDQNModel, policy: BasePolicy,
                 states: TensorType[..., "batch"], next_states: TensorType[..., "batch"],
                 rewards: TensorType[..., "batch"], actions: TensorType[..., "batch"],
                 dones: TensorType[..., "batch"], gamma: float) -> Tuple[
        TensorType[..., "batch", torch.float32], TensorType[..., "batch", torch.float32]]:
        # Max action value for each episode in the sample
        target_values = q_target(next_states)
        greedy_actions = policy.exploit(target_values).unsqueeze(0)
        greedy_value = target_values.gather(-1, greedy_actions)
        
        # Calculate the target action-value for taking each action from each origin state in the
        # sample. If the episode is terminal, the action-value is the reward
        # dones = dones.repeat((1, greedy_value.shape[1])).unsqueeze(2)
        # rewards = rewards.repeat((1, greedy_value.shape[1])).unsqueeze(2)
        target_estimate = rewards + gamma * greedy_value * (1 - dones)
        
        # Get the estimates for the local network and gather the action-value for each action
        # taken in the sample.
        local_estimate = q_local(states)
        actions = actions.unsqueeze(0)
        local_estimate = local_estimate.gather(-1, actions)
        return target_estimate, local_estimate


class DoubleDQNEstimatorStrategy(BaseEstimatorStrategy):
    @staticmethod
    @typechecked
    def estimate(q_target: BaseDQNModel, q_local: BaseDQNModel, policy: BasePolicy,
                 states: TensorType[..., "batch"], next_states: TensorType[..., "batch"],
                 rewards: TensorType[..., "batch"], actions: TensorType[..., "batch"],
                 dones: TensorType[..., "batch"], gamma: float) -> Tuple[
        TensorType[..., "batch", torch.float32], TensorType[ ..., "batch", torch.float32]]:
        local_values = q_local(next_states)
        local_actions = torch.from_numpy(policy.exploit(local_values)).unsqueeze(0).to(local_values.device)
        
        # Calculate the target action-value for taking each action from each origin state in the
        # sample. If the episode is terminal, the action-value is the reward
        #dones = dones.repeat((1, greedy_value.shape[1])).unsqueeze(2)
        #rewards = rewards.repeat((1, greedy_value.shape[1])).unsqueeze(2)
        target_values = q_target(next_states).gather(0, local_actions)
        target_estimate = rewards + gamma * target_values * (1 - dones)
        
        # Get the estimates for the local network and gather the action-value for each action
        # taken in the sample.
        actions = actions.unsqueeze(0)
        local_estimate = q_local(states).gather(0, actions)
        return target_estimate, local_estimate
