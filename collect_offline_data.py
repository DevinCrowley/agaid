from importlib import  import_module
from pathlib import Path
from collections import defaultdict

# import ipdb as pdb #  for debugging
from ipdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from mt_model_buffer import MT_Model_Buffer


def collect_offline_data(env, actor, min_total_steps, max_episode_steps=np.inf, save_path=None):
    actor.eval()

    buffer = MT_Model_Buffer()
    total_steps = 0
    while total_steps < min_total_steps:
        # Sample an episode.
        observation, info = env.reset()
        for step in range(max_episode_steps):
            action = actor(torch.Tensor(observation)).detach().numpy()
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(state=observation,  action=action, next_state=next_observation, done=done)
            if done:
                break
            else:
                observation = next_observation
        buffer.end_trajectory()
        total_steps += step + 1
    # env.close()
    if save_path:
        buffer.save(file_path=save_path, create_dir=True, overwrite_file=True)
    return buffer


if __name__ == '__main__':
    # NOTE: SET
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    env_id = "Pendulum-v1"
    actor_step = 25000
    run_name = "Pendulum-v1__td3_continuous_action__1__1713421739"

    task = None # TODO: integrate this as the task, g for pendulum.

    min_total_steps = 100000
    max_episode_steps = 2000
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    save_path = f"offline_data/{env_id}/td3_step_{actor_step}"
    path_to_cleanrl = Path("../cleanrl").expanduser().resolve()
    actor_path = path_to_cleanrl / f"runs/{run_name}/td3_continuous_action.cleanrl_model_step_{actor_step}"
    assert actor_path.is_file(), f"actor_path does not exist. actor_path:\n{actor_path}"
    
    # from cleanrl.td3_continuous_action.py import Actor as Agent
    # agent_module = import_module(str(path_to_cleanrl / "cleanrl/td3_continuous_action.py"))
    # agent_module = import_module("cleanrl.td3_continuous_action.py", package=path_to_cleanrl)
    # from agent_module import Actor
    ###############################################################
    class Actor(nn.Module):
        def __init__(self, env):
            super().__init__()
            if hasattr(env, 'single_observation_space'): observation_space = env.single_observation_space
            else: observation_space = env.observation_space
            if hasattr(env, 'single_action_space'): action_space = env.single_action_space
            else: action_space = env.action_space
            self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc_mu = nn.Linear(256, np.prod(action_space.shape))
            # action rescaling
            self.register_buffer(
                "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
            )

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias
    ###############################################################

    env = gym.make(env_id, render_mode="human")
    actor = Actor(env)

    buffer = collect_offline_data(env=env, actor=actor, min_total_steps=min_total_steps, max_episode_steps=max_episode_steps, save_path=save_path)