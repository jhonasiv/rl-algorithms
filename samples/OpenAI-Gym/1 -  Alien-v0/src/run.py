import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from torch import nn
from torch.optim import Adam

from model import AlienDQN
from rlalgs.value_based.agent import BaseAgent, make_agent
from rlalgs.value_based.policies import DecayEpsilonGreedy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def train(env: TimeLimit, agent: BaseAgent, num_eps: int):
    cumulative_score = []
    for n in range(num_eps):
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(-1)
        ep_score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_state).float().unsqueeze(-1)
            reward_tensor = torch.tensor([reward]).float()
            done_tensor = torch.tensor([done]).byte()
            agent.step(state, action, reward_tensor, next_state, done_tensor)
            # env.render()
            state = next_state
            ep_score += reward
            if done:
                break
        cumulative_score.append(ep_score)
        print(f'\rEpisode {n}/{num_eps}:\t Average Score: {np.mean(cumulative_score)}\t '
              f'Moving Average (100 last): {np.mean(cumulative_score[-100:])}\tBest Score: ' \
              f'{np.max(cumulative_score)}', end="")


def run(seed, update_every, gamma, tau, lr):
    env = gym.make('Assault-v0')
    action_size = env.action_space.n
    state_space = env.observation_space
    
    policy = DecayEpsilonGreedy(epsilon=1., discrete=True, epsilon_min=0.02,
                                epsilon_decay_rate=0.999)
    model = AlienDQN(
            convolutional_layers=nn.Sequential(nn.Conv2d(3, 4, (5, 5)), nn.ReLU(),
                                               nn.Conv2d(4, 8, (3, 3)), nn.ReLU()),
            linear_layers=nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, action_size)), device=device,
            input_dim=state_space.shape)
    agent = make_agent(seed=seed, update_every=update_every, gamma=gamma, tau=tau, device=device,
                       optimizer_cls=Adam, lr=lr, policy=policy, model=model,
                       replay_buffer_args={"batch_size": 32, "buffer_size": int(1e5)})
    
    train(env, agent, 200)


if __name__ == '__main__':
    run(seed=0, update_every=4, gamma=0.99, tau=1e-3, lr=1e-3)
