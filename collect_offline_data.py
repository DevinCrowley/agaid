from importlib import  import_module
from pathlib import Path
import re
from collections import defaultdict
import time
from dataclasses import dataclass

import tyro
# import ipdb as pdb #  for debugging
from ipdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from mt_model_buffer import MT_Model_Buffer


@dataclass
class Args:
    env_id: str = "Pendulum-v1"
    """The id of the environment"""
    actor_step: int = 25000
    """The number of steps for the actor checkpoints to load"""
    actor_run_name: str = None
    """The exact string for actor_run_name, overwrites actor_run_name_pattern"""
    actor_run_name_pattern: str = "td3_continuous_action__task_g_\d+\.\d+\w*"
    """The regex pattern for each actor_run_name if actor_run_name is not provided"""
    task: float = None
    """the task value for this collection, if not provided it is inferred from the actor_run_name(s)"""
    min_total_steps: int = 100000
    """The minimum number of steps to collect episodes until"""
    max_episode_steps: int = 2000
    """The number of steps per episode to truncate at"""
    seed: int = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""


def collect_offline_data(env, actor, task, min_total_steps, max_episode_steps=np.inf, save_path=None):
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
            buffer.push(task=task, state=observation, action=action, next_state=next_observation, done=done)
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
    args = tyro.cli(Args)
    assert args.task

    path_to_cleanrl = Path("../cleanrl").expanduser().resolve()

    save_path = f"offline_data/{args.env_id}/td3_task_{args.task}_step_{args.actor_step}"
    actor_path = path_to_cleanrl / f"runs/{args.actor_run_name}/td3_continuous_action.cleanrl_model_step_{args.actor_step}"
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


    if args.actor_run_name:
        # CLI specifies single actor_run_name.
        actor_run_names = [args.actor_run_name]
    else:
        # CLI gives regex pattern for actor_run_name strings.
        actor_dir = path_to_cleanrl / f"runs/{args.env_id}"
        assert actor_dir.is_dir(), f"actor_dir does not exist.\nactor_dir:\n {actor_dir}"
        pattern = re.compile(args.actor_run_name_pattern)
        actor_run_names = []
        for actor_path in actor_dir.iterdir():
            if pattern.fullmatch(actor_path.name): actor_run_names.append(actor_path.name)

    for actor_run_name in actor_run_names:
        # Seeding
        # random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Get task.
        if args.task is not None:
            task = args.task
        else:
            task_match = re.search('task_g_(?P<task>\d+(\.\d+)?)')
            assert task_match is not None
            task = float(task_match.group('task'))

        assert args.env_id == 'Pendulum-v1' # env-specific task specification. Note: redo Actor for different policy algo.
        env = gym.make(args.env_id, render_mode="human", g=task)
        actor = Actor(env)

        save_path = path_to_cleanrl / f"offline_data/{args.env_id}/task_{task}/min_total_steps_{args.min_total_steps}__actor__{actor_run_name}"

        start_time = time.monotonic()
        buffer = collect_offline_data(env=env, actor=actor, task=task, min_total_steps=args.min_total_steps, max_episode_steps=args.max_episode_steps, save_path=save_path)
        end_time = time.monotonic()
        print(f"Done. \nTotal steps: {buffer.size} \nDuration: {(end_time - start_time)/3600:.1f} hours")