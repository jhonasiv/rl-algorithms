import random
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Collection, Tuple

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from rlalgs.value_based.annealing import BaseFunction

try:
    from gym.wrappers import LazyFrames
except ImportError:
    LazyFrames = torch.Tensor

Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

patch_typeguard()


@typechecked
def unpack_samples(samples: Tuple[np.ndarray, ...], device: torch.device) -> Tuple[
    TensorType[..., "batch"], TensorType[..., "batch"], TensorType[..., "batch"], TensorType[
        ..., "batch"], TensorType[..., "batch", torch.uint8]]:
    """
    Unpack samples, resulting in torch Tensor with shape (batch_size, *) for each field
    
    :param samples: sampled experiences
    :param device: exec device
    :return: states, actions, rewards, next_states and done samples
    """
    states = torch.from_numpy(np.moveaxis(samples[0], 0, -1)).to(device)
    actions = torch.from_numpy(samples[1]).long().to(device)
    rewards = torch.from_numpy(samples[2]).to(device)
    next_states = torch.from_numpy(np.moveaxis(samples[3], 0, -1)).to(device)
    dones = torch.from_numpy(samples[4]).byte().to(device)
    
    return states, actions, rewards, next_states, dones


class BaseBuffer(ABC):
    def __init__(self, batch_size, buffer_size, seed, device, memmaping: bool = False,
                 memmap_path: str = ''):
        self.memmaping = memmaping
        self.memmap_path = memmap_path
        random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.memory: np.ndarray = None
        self.memory_pos = 0
        self.buffer_length = 0
        self.device = device
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
    
    def _add_to_buffer(self, data: Collection[np.ndarray]) -> None:
        data = [datum.T for datum in data]
        
        data_batch = data[0].shape[0]
        
        memory_slice = self.memory_pos + data_batch
        if memory_slice >= self.buffer_size:
            diff = self.buffer_size - self.memory_pos
            self.memory[self.memory_pos:] = [dat for dat in zip(*(datum[:diff] for datum in data))]
            
            self.memory[:data_batch - diff] = [dat for dat in
                                               zip(*(datum[diff:] for datum in data))]
            
            self.memory_pos = diff
            self.buffer_length = self.buffer_size
        
        else:
            self.memory[self.memory_pos:memory_slice] = [dat for dat in zip(*data)]
            self.memory_pos = memory_slice % self.buffer_size
            if self.buffer_length != self.buffer_size:
                self.buffer_length = self.memory_pos
    
    @abstractmethod
    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
            done: np.ndarray) -> None:
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
        super().__init__(batch_size, buffer_size, seed, device)
        self.memory = np.empty(buffer_size,
                               dtype=[("state", object), ("action", object), ("reward", object),
                                      ("next_state", object), ("done", object)])
    
    @typechecked
    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
            done: np.ndarray) -> None:
        self._add_to_buffer((state, action, reward, next_state, done))
    
    @typechecked
    def sample(self) -> Tuple[
        TensorType[..., "batch"], TensorType[..., "batch"], TensorType[..., "batch"], TensorType[
            ..., "batch"], TensorType[..., "batch", torch.uint8]]:
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
    def update(self, loss: TensorType["batch", torch.float32],
               samples_id: TensorType["batch", torch.long], replay_buffer: Any) -> None:
        """
        Update priorities
        :param loss: TD-difference from last sample
        :param samples_id: list of samples id
        :param replay_buffer: replay buffer instance
        """


@dataclass
class ProportionalUpdateVariant(BaseUpdateVariant):
    constant: float
    
    @typechecked
    def update(self, loss: TensorType["batch", torch.float32],
               samples_id: TensorType["batch", torch.long], priorities: np.ndarray) \
            -> None:
        # priorities[samples_id] = torch.abs(loss) + self.constant
        priorities[samples_id.cpu()] = (torch.abs(loss) + self.constant).detach().cpu().numpy()


class RankedUpdateVariant(BaseUpdateVariant):
    @typechecked
    def update(self, loss: TensorType["batch", torch.float32],
               samples_id: TensorType["batch", torch.long], replay_buffer: Any) -> None:
        loss_sorted_indices = torch.sort(loss).indices
        ranks = 1 / (torch.arange(len(loss_sorted_indices)) + 1)
        sorted_sample_id = samples_id[loss_sorted_indices]
        replay_buffer.memory["priority"][sorted_sample_id] = ranks


class PrioritizedReplayBuffer(BaseBuffer):
    @typechecked
    def __init__(self, batch_size: int, buffer_size: int, seed: int, device: torch.device,
                 alpha: BaseFunction, update_variant: BaseUpdateVariant, memmaping: bool,
                 memmap_path: str):
        """
        Prioritized replay buffer constructor
        
        :param batch_size: size of minibatches
        :param buffer_size: max size of memory
        :param seed: random seed
        :param device: exec device
        :param alpha: prioritization exponent
        :param update_variant: priorities update method
        """
        super().__init__(batch_size, buffer_size, seed, device, memmaping, memmap_path)
        self.update_variant = update_variant
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        
        self.memory_pos = 0
        self.buffer_length = 0
        
        if self.memmaping:
            self.priorities = np.memmap(f'{memmap_path}/priorities.dat', mode='w+',
                                        dtype=np.float32,
                                        shape=(buffer_size,))
            self.rewards = np.memmap(f'{memmap_path}/rewards.dat', mode='w+', dtype=np.float32,
                                     shape=(buffer_size,))
            self.dones = np.memmap(f'{memmap_path}/dones.dat', mode='w+', dtype=np.uint8,
                                   shape=(buffer_size,))
            self.actions = np.memmap(f'{memmap_path}/actions.dat', mode='w+', dtype=np.long,
                                     shape=(buffer_size,))
        else:
            self.priorities = np.zeros((buffer_size,), dtype=np.float32)
            self.rewards = np.zeros((buffer_size,), dtype=np.float32)
            self.dones = np.zeros((buffer_size,), dtype=np.uint8)
            self.actions = np.zeros((buffer_size,), dtype=np.long)
        self.states = None
        self.next_states = None
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
        
        self.alpha = alpha
        self.probabilities = torch.tensor([], requires_grad=False)
    
    def __getitem__(self, item):
        return (self.states[item], self.actions[item], self.rewards[item], self.next_states[
            item], self.dones[item])
    
    def _add_to_buffer(self, data: Collection[np.ndarray]) -> None:
        state, action, reward, next_state, done, priorities = data
        
        data_batch = state.shape[-1]
        
        memory_slice = self.memory_pos + data_batch
        
        def input_experiences(arr, input_data):
            if memory_slice >= self.buffer_size:
                diff = self.buffer_size - self.memory_pos
                arr[self.memory_pos:] = input_data[:diff]
                arr[:data_batch - diff] = input_data[diff:]
            else:
                arr[self.memory_pos:memory_slice] = input_data
        
        state = np.moveaxis(state, -1, 0)
        cached_state = self.next_states[self.memory_pos - data_batch: self.memory_pos]
        if np.all(state == cached_state) and self.memory_pos != 0:
            input_experiences(self.states, cached_state)
        else:
            input_experiences(self.states, state)
        next_state = np.moveaxis(next_state, -1, 0)
        input_experiences(self.rewards, reward)
        input_experiences(self.actions, action)
        input_experiences(self.next_states, next_state)
        input_experiences(self.dones, done)
        input_experiences(self.priorities, priorities)
        
        input_experiences(self.next_states, np.arange(self.memory_pos, memory_slice))
        input_experiences(self.next_states, next_state)
        
        self.memory_pos = memory_slice % self.buffer_size
        self.buffer_length = min(self.buffer_length + data_batch, self.buffer_size)
    
    @typechecked
    def add(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
            done: np.ndarray) -> None:
        if self.buffer_length != 0:
            priority = self.priorities.max()
        else:
            priority = 1
            if self.memmaping:
                self.states = np.memmap(f'{self.memmap_path}/states.dat', dtype=np.uint8, mode='w+',
                                        shape=(self.buffer_size, *state.shape[:-1]))
                self.next_states = np.memmap(f'{self.memmap_path}/next_states.dat',
                                             dtype=np.uint8, mode='w+',
                                             shape=(self.buffer_size, *state.shape[:-1]))
            else:
                self.states = np.zeros((self.buffer_size, *state.shape[:-1]), dtype=state.dtype)
                self.next_states = np.zeros((self.buffer_size, *next_state.shape[:-1]),
                                            dtype=next_state.dtype)
        
        self._add_to_buffer((state, action, reward, next_state, done, np.array([priority])))
    
    @typechecked
    def sample(self) -> Tuple[
        TensorType[..., "batch"], TensorType[..., "batch"], TensorType[..., "batch"], TensorType[
            ..., "batch"], TensorType[..., "batch", torch.uint8], TensorType[
            ..., "batch", torch.long]]:
        p_i = torch.from_numpy(self.priorities[:self.buffer_length]).to(self.device).pow(
                self.alpha.value())
        self.probabilities = (p_i / torch.sum(p_i))
        
        samples_id = torch.multinomial(self.probabilities, self.batch_size, replacement=False)
        samples = self[samples_id.cpu()]
        
        states, actions, rewards, next_states, dones = unpack_samples(samples, self.device)
        
        return states, actions, rewards, next_states, dones, samples_id
    
    @typechecked
    def calc_is_weight(self, samples_id: TensorType["batch", torch.long], beta: float) -> \
            TensorType["batch", torch.float32]:
        """
        Calculate Importance-Sampling weights for bias correction
        :param beta: importance-sampling bias exponent
        :param samples_id: list of IDs for each sampled experience in this time step
        :return: IS weights
        """
        sample_probabilities = self.probabilities[samples_id]
        weights: torch.Tensor = ((1 / (len(self) * sample_probabilities)) ** beta)
        weights = weights / weights.max()
        return weights.float().to(self.device)
    
    @typechecked
    def update(self, loss: TensorType[-1, torch.float32],
               samples_id: TensorType[-1, torch.long]) -> None:
        """
        Update priorities for every experience
        
        :param loss: TD-difference from the last gradient descent step
        :param samples_id: list of IDs for each sampled experience in this time step
        """
        priorities = self.priorities
        self.update_variant.update(loss, samples_id, priorities)
    
    @typechecked
    def __len__(self) -> int:
        return self.buffer_length
