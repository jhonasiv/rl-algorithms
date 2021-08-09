import argparse

import gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD

from model import AlienDQN
from rlalgs.utils import gym_wrappers
from rlalgs.value_based.agent import BaseAgent, make_agent
from rlalgs.value_based.annealing import Constant, LinearAnnealing
from rlalgs.value_based.policies import LinearAnnealedEpsilon
from rlalgs.value_based.replay import ProportionalUpdateVariant


def train(env: gym.Env, agent: BaseAgent, num_steps: int, render: bool):
    cumulative_score = []
    steps = 0
    episodes = 0
    while True:
        state = env.reset()
        ep_score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, np.array([reward]), next_state, np.array([done]))
            if render:
                env.render()
            state = next_state
            ep_score += reward
            steps += 1
            if done or steps >= num_steps:
                break
        cumulative_score.append(ep_score)
        episodes += 1
        print(f'Steps {steps}/{num_steps}:\t Last Score: {ep_score}\t '
              f'Moving Average (100 last eps.): {np.mean(cumulative_score[-100:]):.2f}\tBest '
              f'Score: {np.max(cumulative_score)}')
        if steps >= num_steps:
            break
    agent.save("../ckpt/ckpt.pth")


def run(seed, update_every, gamma, tau, lr, batch_size, render, gpu, buffer_size,
        learning_threshold, env, num_steps, sampling_rate):
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if gpu == 'yes' else "cpu")
    
    env = gym.make(env)
    env = gym_wrappers.MaxAndSkipEnv(env, skip=4)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 110))
    env = gym_wrappers.CropObservation(env, slices=(slice(None, None), slice(18, 102),))
    # env = gym.wrappers.TransformObservation(env, lambda x: x / 255.)
    env = gym_wrappers.TransposeObservation(env)
    env = gym_wrappers.StackFrame(env, stacks=4, shape=(4, 84, 84), stack_axis=0)
    env = gym.wrappers.TransformObservation(env, lambda x: x.reshape(*x.shape, 1))
    action_size = env.action_space.n
    state_space = env.observation_space
    
    input_dim = (84, 84, 4)
    
    policy = LinearAnnealedEpsilon(epsilon=1., discrete=True, epsilon_min=0.1, num_steps=int(1e6))
    # policy = ConstantEpsilonGreedy(epsilon=0.01, discrete=True)
    model = AlienDQN(
            convolutional_layers=nn.Sequential(nn.Conv2d(input_dim[-1], 32, (8, 8), stride=(4, 4)),
                                               nn.ReLU(), nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
                                               nn.ReLU(), nn.Conv2d(64, 64, (3, 3)), nn.ReLU()),
            linear_layers=[nn.Linear(512, action_size), ], device=device,
            input_dim=input_dim)
    agent = make_agent(seed=seed, update_every=update_every, gamma=gamma, tau=tau, device=device,
                       double_dqn=True, sampling_rate=sampling_rate, optimizer_cls=Adam, lr=lr,
                       learning_threshold=learning_threshold, policy=policy,
                       prioritized_replay_buffer=True, model=model,
                       replay_buffer_args={"batch_size"    : batch_size,
                                           "buffer_size"   : int(buffer_size),
                                           "alpha"         : Constant(0.6),
                                           "beta"          : LinearAnnealing(0.4, 1, num_steps),
                                           "update_variant": ProportionalUpdateVariant(1e-6)})
    
    train(env, agent, num_steps, render)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--update_every", default=4, type=int, help="DQN update rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="return discount rate")
    parser.add_argument("--tau", default=1e-3, type=float, help="step-size constant target network "
                                                                "soft update")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--render", nargs='?', const=True, default=False, help="render env")
    parser.add_argument("--gpu", default=None, choices=[None, 'no', 'yes'], help="use gpu")
    parser.add_argument("--buffer_size", default=1e5, type=float,
                        help="maximum size of the replay buffer")
    parser.add_argument("--learning_threshold", default=int(5e4), type=int,
                        help="number of time steps before the learning starts")
    parser.add_argument("--env", default="Assault-v0", type=str, help="gym env to run")
    parser.add_argument("--num_steps", default=50e6, type=int, help="number of steps to run")
    parser.add_argument("--sampling_rate", default=1, type=int, help="rate to update local network")
    
    args = parser.parse_args()
    run(**vars(args))
