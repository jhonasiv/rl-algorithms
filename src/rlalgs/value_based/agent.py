import abc
from dataclasses import dataclass, field
from typing import Any, Collection, Type, Union

import numpy as np
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from rlalgs.value_based.policies import BaseEpsilonGreedyPolicy
from rlalgs.value_based.strategy.learning import (BaseLearningStrategy, DQNLearningStrategy,
                                                  PrioritizedLearningStrategy)
from .model import BaseDQNModel
from .replay import BaseBuffer, Experience
from .strategy.estimate import (BaseEstimatorStrategy, DQNEstimatorStrategy,
                                DoubleDQNEstimatorStrategy)

patch_typeguard()

torch.autograd.set_detect_anomaly(True)

@dataclass
class BaseAgent(abc.ABC):
    """
    Base deep reinforcement learning agent.
    """
    seed: int  # random seed
    device: torch.device  # torch device
    update_every: int = 4  # how often the target network is updated
    gamma: float = 0.99  # discount factor
    tau: float = 1e-3  # step-size for soft updating the target network
    learning_threshold: int = 0  # after which step to start learning
    time_step: int = field(init=False, default=0)  # current time step
    batch_size: int = field(init=False, default=64)  # experience memory batch size
    q_local: BaseDQNModel = field(init=False, default=None)  # local/online network
    q_target: BaseDQNModel = field(init=False, default=None)  # target network
    estimation_strategy: BaseEstimatorStrategy = field(init=False,
                                                       default=None)  # estimates the action
    # values for each step
    learning_strategy: BaseLearningStrategy = field(init=False,
                                                    default=None)  # strategy to learn from
    # estimates
    memory: BaseBuffer = field(init=False, default=None)  # experience replay buffer
    policy: BaseEpsilonGreedyPolicy = field(init=False, default=None)  # policy
    optimizer: torch.optim.Optimizer = field(init=False, default=None)  # torch optimizer
    episode: int = field(init=False, default=0)  # number of episodes
    
    def __post_init__(self):
        self.rng = np.random.Generator(np.random.PCG64(seed=self.seed))
    
    @typechecked
    def load(self, path: str) -> None:
        """
        Load previously trained model.
        
        :param path: model path
        """
        self.q_local.load_state_dict(torch.load(path))
    
    @typechecked
    def save(self, path: str) -> None:
        """
        Save trained model.
        
        :param path: saved model path
        """
        torch.save(self.q_local.parameters(), path)
    
    @typechecked
    def set_optimizer(self, optimizer_cls: Type[torch.optim.Optimizer], lr: float) -> None:
        """
        Sets the training optimizer
        
        :param optimizer_cls: new training optimizer class
        :param lr: learning rate
        """
        self.optimizer = optimizer_cls(self.q_local.parameters(), lr)
    
    @typechecked
    def set_model(self, model: BaseDQNModel) -> None:
        """
        Sets the model for local and target network
        
        :param model: neural network model
        """
        self.q_local = model
        self.q_target = model.copy()
    
    @typechecked
    def set_estimator(self, estimator: BaseEstimatorStrategy) -> None:
        """
        Sets the action values estimator object
        :param estimator: new estimator object
        """
        self.estimation_strategy = estimator
    
    @typechecked
    def set_learning_strategy(self, learning_strategy: BaseLearningStrategy) -> None:
        """
        Sets the learning strategy object
        
        :param learning_strategy: subclass of BaseLearningStrategy
        """
        self.learning_strategy = learning_strategy
        self.memory = self.learning_strategy.memory
        self.batch_size = self.memory.batch_size
    
    @typechecked
    def set_policy(self, policy: BaseEpsilonGreedyPolicy) -> None:
        """
        Sets the policy.
        :param policy: new epsilon greedy policy
        """
        self.policy = policy
    
    @abc.abstractmethod
    def step(self, state: Collection[float], action: int, reward: float,
             next_state: Collection[float], done: bool) -> None:
        """
        The agent registers a step it took in the environment.

        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: state that resulted from the action
        :param done: if the resulting state was terminal
        """
    
    @abc.abstractmethod
    def learn(self, experiences: Collection[Experience]) -> None:
        """
        The agent learns from previous experiences.
        :param experiences: a minibatch of previous experiences with size=batch_size
        """
    
    @abc.abstractmethod
    def act(self, state: TensorType[..., "batch"], train: bool = True) -> Union[
        int, float, torch.Tensor, np.ndarray]:
        """
        The agent acts following a epsilon-greedy policy.
        :param train: if training mode is active, if so the agent will follow the epsilon-greedy
        policy, otherwise it
        will follow the greedy policy
        :param state: current state
        :return: action selected
        """
    
    @abc.abstractmethod
    def soft_update(self) -> None:
        """
        Soft-updates the target network with the recently-updated parameters of the local
        network, with self.tau as
        step-size.
        """


@dataclass
class DQNetAgent(BaseAgent):
    """
    Deep Q-Network agent.
    """
    
    def __init__(self, **data: Any):
        super(DQNetAgent, self).__init__(**data)
    
    @typechecked
    def step(self, states: TensorType[..., "batch"], actions: TensorType[..., "batch"],
             rewards: TensorType[..., "batch"], next_states: TensorType[..., "batch"],
             done: TensorType[..., "batch", torch.uint8]) -> None:  # Store experience in memory
        
        self.memory.add(states.to(self.device), actions.to(self.device), rewards.to(self.device),
                        next_states.to(self.device), done.to(self.device))
        
        if torch.any(done):
            self.episode += 1
        # Learn every update_every time steps
        self.time_step += states.shape[-1]
        if self.time_step > self.learning_threshold:
            # Check if there are enough samples in memory, if so, get a sample and learn from it
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    @typechecked
    def learn(self, experiences: Collection[Experience]) -> None:
        self.learning_strategy.learn(experiences=experiences, policy=self.policy,
                                     estimation_strategy=self.estimation_strategy,
                                     q_target=self.q_target, q_local=self.q_local,
                                     optimizer=self.optimizer, gamma=self.gamma,
                                     episode=self.episode)
        
        # Update the target network
        if self.time_step % self.update_every == 0:
            self.soft_update()
    
    @typechecked
    def act(self, states: TensorType[..., "batch"], train=True) -> TensorType[..., "batch"]:
        epsilon = self.policy.step(self.time_step)
        states = states.to(self.device)
        
        # Get estimate action values from local network
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(states)
        self.q_local.train()
        
        # Epsilon-greedy action selection
        if train:
            if self.rng.random() > epsilon:
                return self.policy.exploit(action_values)
            else:
                return self.policy.explore(action_values)
        else:
            return self.policy.exploit(action_values)
    
    @typechecked
    def soft_update(self) -> None:
        for target_p, local_p in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_p.data.copy_(self.tau * local_p.data + (1.0 - self.tau) * target_p.data)


class InvalidParameters(Exception):
    def __init__(self, message):
        self.message = message


def make_agent(seed: int, update_every: int, gamma: float, tau: float, device: torch.device,
               learning_threshold: int, optimizer_cls: Type[torch.optim.Optimizer], lr: float,
               policy: BaseEpsilonGreedyPolicy, model: BaseDQNModel, double_dqn: bool = False,
               prioritized_replay_buffer: bool = False, replay_buffer_args: dict = None) -> \
        DQNetAgent:
    """
    This function yields an agent object with the selected input parameters.

    ----
    
    Experience Replay Buffer specific parameters:
        - batch-size (int): size of the minibatches sampled from replay buffer. Default to 64.
        - buffer-size (int): maximum size of the replay buffer. Defaults to 1e5.
        - alpha (float): prioritization exponent. (Prioritized Experience Buffer)
        - beta (AbstractBetaFunction): importance-sampling exponent. (Prioritized Experience Buffer)
        - update_variant (AbstractUpdateVariant): replay buffer priority update variant. (
        Prioritized Experience Buffer)
    
    
    
    :param learning_threshold: which time step to start learning
    :param device:  device to host tensors
    :param state_size: size of each state in the state space
    :param action_size: size of the action space
    :param seed: random seed
    :param update_every: how often the target network is updated
    :param gamma: discount rate
    :param tau: step-size for soft updating the target network
    :param optimizer_cls: type of training optimizer
    :param lr: learning rate
    :param policy: action selection policy
    :param model: instance of neural networks model
    :param double_dqn: will the agent use the Double DQN estimation strategy
    :param prioritized_replay_buffer: will the agent use a Prioritized Experience Replay buffer
    

    :keyword replay_buffer_args: see Replay Buffer specific parameters section.
    :keyword networks_args: see Neural network specific parameters section.
    :return:
    """
    
    if prioritized_replay_buffer and set(replay_buffer_args.keys()) != {"batch_size", "buffer_size",
                                                                        "alpha", "beta",
                                                                        "update_variant"}:
        raise InvalidParameters(
                "When using prioritized replay buffer, replay_buffer_args should contain the "
                "following keys: "
                f"{'batch_size', 'buffer_size', 'alpha', 'beta', 'update_variant'}. It contains "
                f"{set(replay_buffer_args.keys())}")
    if replay_buffer_args is None:
        replay_buffer_args = {"batch_size": 64, "buffer_size": int(1e5)}
    options_mapping = {
            'double_dqn'               : {True : DoubleDQNEstimatorStrategy,
                                          False: DQNEstimatorStrategy},
            'prioritized_replay_buffer': {True : PrioritizedLearningStrategy,
                                          False: DQNLearningStrategy}, }
    
    agent = DQNetAgent(seed=seed, update_every=update_every, gamma=gamma, tau=tau, device=device,
                       learning_threshold=learning_threshold)
    estimator = options_mapping['double_dqn'][double_dqn]
    agent.set_estimator(estimator())
    
    learning_strategy = options_mapping['prioritized_replay_buffer'][prioritized_replay_buffer]
    agent.set_learning_strategy(learning_strategy(**replay_buffer_args, seed=seed, device=device))
    
    agent.set_model(model)
    
    agent.set_optimizer(optimizer_cls=optimizer_cls, lr=lr)
    
    agent.set_policy(policy)
    return agent
