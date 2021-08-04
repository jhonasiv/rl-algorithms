import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple, TerminalSteps
from mlagents_envs.environment import UnityEnvironment
from torch import nn

from unity.model import DuelingBallModel
from utils.grid import Grid
from value_based.agent import DQNetAgent, make_agent
from value_based.policies import DecayEpsilonGreedy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(env_filename: str, seed: int, update_rate: int, lr: float, tau: float, gamma: float,
        num_tilings: int, batch_size: int):
    env = UnityEnvironment(env_filename, seed=seed)
    env.reset()
    
    behavior_specs_name = list(env.behavior_specs.keys())[0]
    action_size = env.behavior_specs[behavior_specs_name].action_spec.continuous_size
    state_size = env.behavior_specs[behavior_specs_name].observation_specs[0].shape[0]
    
    action_grids = Grid(lower_bounds=[-1, -1], upper_bounds=[1, 1],
                        n_tilings=[num_tilings, num_tilings])
    
    policy = DecayEpsilonGreedy(epsilon=0.9, discrete=True, epsilon_min=0.01,
                                epsilon_decay_rate=0.995)
    # model = BallModel(sequential_model=nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(),
    #                                                  nn.Linear(128, 256), nn.ReLU(),
    #                                                  nn.Linear(256, 1024), nn.ReLU()),
    #                   rotation_x_layer=nn.Sequential(nn.Linear(1024, num_tilings)),
    #                   rotation_z_layer=nn.Sequential(nn.Linear(1024, num_tilings)), device=device)
    model = DuelingBallModel(hidden_sequential=nn.Sequential(nn.Linear(state_size, 128),
                                                             nn.ReLU(), nn.Linear(128, 512),
                                                             nn.ReLU(), nn.Linear(512, 1024),
                                                             nn.ReLU()),
                             value_sequential=nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                            nn.Linear(512, 1), nn.ReLU()),
                             advantage_sequential=nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                                                nn.Linear(512, num_tilings),
                                                                nn.ReLU()),
                             rotation_x_layer=nn.Sequential(nn.Linear(num_tilings, num_tilings)),
                             rotation_z_layer=nn.Sequential(nn.Linear(num_tilings, num_tilings)),
                             device=device, seed=0)
    agent = make_agent(state_size=state_size, action_size=action_size, seed=seed, model=model,
                       update_every=update_rate, policy=policy, optimizer_cls=torch.optim.Adam,
                       lr=lr, tau=tau, gamma=gamma, double_dqn=True,
                       replay_buffer_args={"batch_size": batch_size, "buffer_size": int(1e5)})
    
    train(env, behavior_specs_name, agent, action_grids, 2000)


def train(env: UnityEnvironment, spec_name: str, agent: DQNetAgent, action_tilings: Grid,
          num_eps: int):
    for j in range(1, num_eps + 1):
        env.reset()
        env.step()
        decision_steps, terminal_steps = env.get_steps(spec_name)
        states = torch.from_numpy(decision_steps.obs[0]).float().to(device)
        episode_rewards = []
        while True:
            terminal = TerminalSteps([], np.array([], np.float32), np.array([]),
                                     np.array([], dtype=np.int32), np.array([]), np.array([]))
            discrete_actions = agent.act(states, train=True)
            continuous_actions = action_tilings.batch_to_continuous(discrete_actions)
            env.set_actions(spec_name, ActionTuple(continuous=continuous_actions.cpu().numpy()))
            env.step()
            decision_steps, terminal_steps = env.get_steps(spec_name)
            if len(decision_steps) > 0:
                terminal = terminal_steps
            while len(decision_steps) == 0:
                if len(terminal.obs) == 0:
                    terminal.obs = terminal_steps.obs
                else:
                    terminal.obs[0] = np.concatenate((terminal.obs[0], terminal_steps.obs[0]))
                terminal.reward = np.concatenate((terminal.reward, terminal_steps.reward))
                terminal.agent_id = np.concatenate((terminal.agent_id, terminal_steps.agent_id),
                                                   dtype=np.int32)
                env.step()
                decision_steps, terminal_steps = env.get_steps(spec_name)
            next_states = torch.from_numpy(decision_steps.obs[0]).float()
            rewards = torch.from_numpy(decision_steps.reward)
            done = torch.full_like(rewards, fill_value=False).byte()
            terminal_id = terminal.agent_id
            if terminal_id.size != 0:
                terminal_agent_idx = np.array([terminal.agent_id_to_index[a] for a in terminal_id])
                next_states[terminal_id] = torch.from_numpy(terminal.obs[0][terminal_agent_idx])
                rewards[terminal_id] = torch.from_numpy(terminal.reward[terminal_agent_idx])
                done[terminal_id] = torch.tensor([True for _ in terminal_id]).byte()
            
            rewards = rewards.unsqueeze(1)
            done = done.unsqueeze(1)
            agent.step(states=states, actions=discrete_actions, next_states=next_states,
                       rewards=rewards, done=done)
            states = next_states


if __name__ == '__main__':
    run(env_filename='/home/jhonas/applications/ml-agents/Project/3dball', seed=0, update_rate=8,
        lr=1e-3, tau=2e-2, gamma=0.99, num_tilings=2048, batch_size=2048)
