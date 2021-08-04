from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import random
import torch
from abc import ABC, abstractmethod
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Any, Collection, Tuple

Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

patch_typeguard()


@typechecked
def unpack_samples(samples: np.ndarray, device: torch.device) -> Tuple[
    TensorType["batch", -1], TensorType["batch", -1], TensorType["batch", -1], TensorType[
        "batch", -1], TensorType["batch", -1]]:
    """
    Unpack samples, resulting in torch Tensor with shape (batch_size, *) for each field
    
    :param samples: sampled experiences
    :param device: exec device
    :return: states, actions, rewards, next_states and done samples
    """
    states = torch.stack(tuple(samples["state"])).to(device)
    actions = torch.stack(tuple(samples["action"])).long().to(device)
    rewards = torch.stack(tuple(samples["reward"])).to(device)
    next_states = torch.stack(tuple(samples["next_state"])).to(device)
    dones = torch.stack(tuple(samples["done"])).byte().to(device)
    
    return states, actions, rewards, next_states, dones


class BaseBuffer(ABC):
    batch_size: int
    memory: np.ndarray
    device: torch.device
    
    @abstractmethod
    def add(self, state: TensorType[-1, -1], action: TensorType[-1, -1], reward: TensorType[-1, -1],
            next_state: TensorType[-1, -1], done: TensorType[-1, -1, torch.uint8]) -> None:
        """
        Add a new experience to memory.
        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: resulting state after action
        :param done: if the resulting state is terminal
        """
    
    @abstractmethod
    def sample(self) -> Collection[Experience]:
        """
        Randomly sample a batch of experiences from memory.
        :return: a minibatch of experiences
        """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the current size of internal memory"""


class ReplayBuffer(BaseBuffer):
    """
    Buffer for stores experiences and sampling them when requested.
    """
    
    @typechecked
    def __init__(self, batch_size: int, buffer_size: int, seed: int, device: torch.device):
        """
        ReplayBuffer constructor
        :param batch_size: number of experiences in a minibatch
        :param buffer_size: max len of memory
        :param seed: random seed
        :param device: torch device
        """
        super(ReplayBuffer, self).__init__()
        random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.memory_pos = 0
        self.buffer_length = 0
        self.memory = np.empty(buffer_size,
                               dtype=[("state", object), ("action", object), ("reward", object),
                                      ("next_state", object), ("done", object)])
        self.device = device
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
    
    @typechecked
    def add(self, state: TensorType["batch", -1], action: TensorType["batch", -1],
            reward: TensorType["batch", -1], next_state: TensorType["batch", -1],
            done: TensorType["batch", -1, torch.uint8]) -> None:
        state = state.to(self.device)
        action = action.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        memory_slice = self.memory_pos + state.shape[0]
        if memory_slice > self.buffer_size:
            diff = self.buffer_size - self.memory_pos
            self.memory[self.memory_pos:] = [(s, a, r, ns, d) for s, a, r, ns, d in
                                             zip(state[:diff], action[:diff], reward[:diff],
                                                 next_state[:diff], done[:diff])]
            
            self.memory[:state.shape[0] - diff] = [(s, a, r, ns, d) for s, a, r, ns, d in
                                                   zip(state[diff:], action[diff:], reward[diff:],
                                                       next_state[diff:], done[diff:])]
            
            self.memory_pos = diff
            self.buffer_length = self.buffer_size
        
        else:
            self.memory[self.memory_pos:memory_slice] = [(s, a, r, ns, d) for s, a, r, ns, d in
                                                         zip(state, action, reward, next_state,
                                                             done)]
            self.memory_pos = memory_slice % self.buffer_size
            if self.buffer_length != self.buffer_size:
                self.buffer_length = self.memory_pos
    
    @typechecked
    def sample(self) -> Tuple[
        TensorType["batch", -1], TensorType["batch", -1], TensorType["batch", -1], TensorType[
            "batch", -1], TensorType["batch", -1]]:
        samples_id = self.rng.choice(len(self), size=self.batch_size)
        samples = self.memory[samples_id]
        
        states, actions, rewards, next_states, dones = unpack_samples(samples, self.device)
        
        return states, actions, rewards, next_states, dones
    
    @typechecked
    def __len__(self) -> int:
        """Return the current size of internal memory"""
        return self.buffer_length


class BaseUpdateVariant(ABC):
    @abstractmethod
    @typechecked
    def update(self, loss: TensorType["loss", torch.float32],
               samples_id: TensorType["ids", torch.int], replay_buffer: Any) -> None:
        """
        Update priorities
        :param loss: TD-difference from last sample
        :param samples_id: list of samples id
        :param replay_buffer: replay buffer instance
        """


@dataclass
class ConstantUpdateVariant(BaseUpdateVariant):
    constant: float
    
    @typechecked
    def update(self, loss: TensorType["loss", torch.float32],
               samples_id: TensorType["ids", torch.int], replay_buffer: Any) -> None:
        for sample_id in samples_id:
            replay_buffer.priorities[sample_id] = loss + self.constant


class RankedUpdateVariant(BaseUpdateVariant):
    @typechecked
    def update(self, loss: TensorType["loss", torch.float32],
               samples_id: TensorType["ids", torch.int], replay_buffer: Any) -> None:
        loss_sorted_indices = torch.sort(loss).indices
        ranks = 1 / (torch.arange(len(loss_sorted_indices)) + 1)
        sorted_sample_id = samples_id[loss_sorted_indices]
        replay_buffer.memory["priority"][sorted_sample_id] = ranks


class PrioritizedReplayBuffer(BaseBuffer):
    @typechecked
    def __init__(self, batch_size: int, buffer_size: int, seed: int, device: torch.device,
                 alpha: float, update_variant: BaseUpdateVariant):
        """
        Prioritized replay buffer constructor
        
        :param batch_size: size of minibatches
        :param buffer_size: max size of memory
        :param seed: random seed
        :param device: exec device
        :param alpha: prioritization exponent
        :param update_variant: priorities update method
        """
        super(PrioritizedReplayBuffer, self).__init__()
        self.update_variant = update_variant
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        
        self.memory_pos = 0
        self.buffer_length = 0
        self.memory = np.empty(buffer_size, dtype=[("priority", np.float), ("state", np.ndarray),
                                                   ("action", torch.Tensor), ("reward", np.float),
                                                   ("next_state", np.ndarray), ("done", np.uint8)])
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
        
        self.alpha = alpha
        self.probabilities = torch.tensor([])
    
    @typechecked
    def add(self, state: Collection[float], action: TensorType["action", "value"],
            reward: np.float32, next_state: Collection[float], done: bool) -> None:
        if self.memory_pos != 0:
            priority = self.memory["priority"].max()
        else:
            priority = 1
        
        self.memory[self.memory_pos] = (priority, state, action, reward, next_state, done)
        self.memory_pos = (self.memory_pos + 1) % self.buffer_size
        self.buffer_length = self.buffer_length + 1 if self.buffer_length < self.buffer_size else \
            self.buffer_size
    
    @typechecked
    def sample(self) -> Tuple[
        TensorType["batch", -1], TensorType["batch", -1], TensorType["batch", -1], TensorType[
            "batch", -1], TensorType["batch", -1], TensorType[-1, torch.int]]:
        p_i = torch.from_numpy(self.memory["priority"][:self.buffer_length].copy()) ** self.alpha
        self.probabilities = (p_i / torch.sum(p_i))
        
        samples_id = torch.from_numpy(
                self.rng.choice(len(self), size=self.batch_size, p=self.probabilities,
                                replace=False)).long()
        samples = self.memory[samples_id]
        
        states, actions, rewards, next_states, dones = unpack_samples(samples, self.device)
        
        return states, actions, rewards, next_states, dones, samples_id
    
    @typechecked
    def calc_is_weight(self, samples_id: TensorType["ids", torch.int], beta: float) -> TensorType[
        "weights", torch.float32]:
        """
        Calculate Importance-Sampling weights for bias correction
        :param beta: importance-sampling bias exponent
        :param samples_id: list of IDs for each sampled experience in this time step
        :return: IS weights
        """
        sample_probabilities = self.probabilities[samples_id]
        weights: torch.Tensor = ((1 / (len(self) * sample_probabilities)) ** beta).refine_names(
                "weights")
        weights /= weights.max()
        return weights.float().to(self.device)
    
    @typechecked
    def update(self, loss: TensorType["loss", torch.float32],
               samples_id: TensorType["ids", torch.int]) -> None:
        """
        Update priorities for every experience
        
        :param loss: TD-difference from the last gradient descent step
        :param samples_id: list of IDs for each sampled experience in this time step
        """
        self.update_variant.update(loss, samples_id, self)
    
    @typechecked
    def __len__(self) -> int:
        return self.buffer_length
