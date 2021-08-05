from collections import deque
from time import sleep

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from model import BasicModel
from torch import nn

from rlalgs.value_based.agent import DQNetAgent, make_agent
from rlalgs.value_based.policies import DecayEpsilonGreedy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(env_filename: str, seed: int, update_rate: int, lr: float, tau: float, gamma: float,
        batch_size: int):
    env = UnityEnvironment(env_filename, seed=seed)
    env.reset()
    
    behavior_specs_name = list(env.behavior_specs.keys())[0]
    action_size = env.behavior_specs[behavior_specs_name].action_spec.discrete_branches[0]
    state_size = env.behavior_specs[behavior_specs_name].observation_specs[0].shape[0]
    
    policy = DecayEpsilonGreedy(epsilon=1., discrete=True, epsilon_min=0.001,
                                epsilon_decay_rate=0.9995)
    
    model = BasicModel(sequential_model=nn.Sequential(nn.Linear(state_size, 64), nn.ReLU(),
                                                      nn.Linear(64, action_size)))
    agent = make_agent(seed=seed, model=model, update_every=update_rate, policy=policy,
                       optimizer_cls=torch.optim.Adam, lr=lr, tau=tau, gamma=gamma, double_dqn=True,
                       replay_buffer_args={"batch_size": batch_size, "buffer_size": int(1e5)})
    
    train(env, behavior_specs_name, agent, 2000)


def train(env: UnityEnvironment, spec_name: str, agent: DQNetAgent, num_eps: int):
    cumulative_rewards = []
    moving_avg = deque(maxlen=100)
    for j in range(1, num_eps + 1):
        env.reset()
        env.step()
        decision_steps, terminal_steps = env.get_steps(spec_name)
        states = torch.from_numpy(decision_steps.obs[0]).float().to(device).T
        episode_rewards = []
        while True:
            discrete_actions = agent.act(states, train=True)
            env.set_actions(spec_name, ActionTuple(discrete=discrete_actions.cpu().numpy()))
            env.step()
            decision_steps, terminal_steps = env.get_steps(spec_name)
            next_states = torch.from_numpy(decision_steps.obs[0]).float()
            rewards = torch.from_numpy(decision_steps.reward)
            if len(terminal_steps) != 0:
                next_states = torch.from_numpy(terminal_steps.obs[0]).float()
                rewards = torch.from_numpy(terminal_steps.reward)
            rewards = rewards.unsqueeze(1)
            done = torch.full_like(rewards, fill_value=len(terminal_steps) != 0).byte()
            agent.step(states=states, actions=discrete_actions, next_states=next_states,
                       rewards=rewards, done=done)
            states = next_states
            episode_rewards.append(rewards.item())
            if len(terminal_steps) != 0:
                break
        cumulative_rewards.append(np.sum(episode_rewards))
        moving_avg.append(np.sum(episode_rewards))
        print(f'\rEpisode {j}/{num_eps} - Average Score: {np.mean(cumulative_rewards)}', end="")
        if j % 100 == 0:
            print(f'\rEpisode {j}/{num_eps} - Moving Score: {np.mean(moving_avg)}\n', end="")
        if np.mean(moving_avg) > 0.93:
            print(f"Achieved moving average of {np.mean(moving_avg)}! Stop training!")
            break
    eval_rewards = []
    for n in range(200):
        env.reset()
        ep_rewards = []
        while True:
            sleep(0.1)
            env.step()
            decision_steps, terminal_steps = env.get_steps(spec_name)
            if len(terminal_steps) == 0:
                ep_rewards.append(decision_steps.reward[0])
                states = torch.from_numpy(decision_steps.obs[0]).float().to(device)
                action = agent.act(states, train=False)
                env.set_actions(spec_name, action=ActionTuple(discrete=action.cpu().numpy()))
            else:
                ep_rewards.append(terminal_steps.reward[0])
                break
        eval_rewards.append(np.sum(ep_rewards))
        print(f"\rEpisode {n}/200! Eval Score: {np.mean(eval_rewards)}")


if __name__ == '__main__':
    run(env_filename='/home/jhonas/applications/ml-agents/Project/Basic', seed=0, update_rate=4,
        lr=5e-3, tau=1e-2, gamma=0.9, batch_size=64)
