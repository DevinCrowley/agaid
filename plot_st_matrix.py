import pickle
from importlib import  import_module
from pathlib import Path
import re
from collections import defaultdict
import time
from copy import deepcopy
from dataclasses import dataclass

import tyro
# import ipdb as pdb #  for debugging
# from ipdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ray

from nn.world_model import Dynamics_Model_Multihead, Dynamics_Model_Embed, Dynamics_Model_Aggregate
from mt_model_buffer import MT_Model_Buffer


@dataclass
class Args:
    env_id: str = "Pendulum-v1"
    """The id of the environment"""
    exp_id: str = "200k_actors_mt_st"
    """The id of the experiment"""
    actor_step: int = None
    """The number of steps for the actor checkpoints to load"""
    actor_run_name: str = None
    """The exact string for actor_run_name, overwrites actor_run_name_pattern"""
    actor_run_name_pattern: str = "td3_continuous_action__task_g_\d+\.\d+\w*"
    """The regex pattern for each actor_run_name if actor_run_name is not provided"""
    min_total_steps: int = 1000
    """The minimum number of steps to collect episodes until"""
    max_episode_steps: int = 2000
    """The number of steps per episode to truncate at"""
    seed: int = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    # cuda: bool = True
    # """if toggled, cuda will be enabled by default"""
    noisy_actions: bool = True
    """whether to add exploration noise to the actions"""
    
    num_workers: int = 1
    """the number of ray workers"""
    # num_envs: int = 1
    # """the number of parallel game environments"""
    # buffer_size: int = int(1e6)
    # """the replay memory buffer size"""
    # policy_noise: float = 0.2
    # """the scale of policy noise"""
    # exploration_noise: float = 0.1
    # """the scale of exploration noise"""


def eval_st_cross_task(env, actor, model, actor_env_task, model_task, buffer, min_total_steps, max_episode_steps=np.inf, noisy_actions=False, save_path=None):
    actor.eval()
    model.eval()

    total_steps = 0
    # print(f"\nStarting collection.") # debug
    mses = []
    while total_steps < min_total_steps:
        # print(f"\nStarting episode:") # debug
        # Sample an episode.
        observation, info = env.reset()
        # print(f"NEW EPISODE") # debug
        for step in range(max_episode_steps):
            # if step % 100 == 0: print(f"Step: {step}") # debug
            action = actor(torch.Tensor(observation), noisy=noisy_actions).detach().numpy()
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            normed_obs = torch.tensor(buffer.normalize_state(observation))
            normed_next_obs = torch.tensor(buffer.normalize_state(next_observation))
            normed_action = torch.tensor(buffer.normalize_action(action))
            pred_normed_next_obs = normed_obs + model(torch.cat([normed_obs, normed_action]))
            mse = (pred_normed_next_obs - normed_next_obs).pow(2).mean()
            mses.append(mse)
            if done:
                # print(f"Episode done -- terminated: {terminated}, truncated: {truncated}") # debug
                break
            else:
                observation = next_observation
        total_steps += step + 1
    # print(f"Done sampling total_steps {total_steps} / min_total_steps {min_total_steps}") # debug
    # print(f"buffer.size: {buffer.size}, len(buffer.states): {len(buffer.states)}, buffer manual size: {sum((len(state_trj) for state_trj in buffer.states))}") # debug
    # env.close()
    if save_path:
        # print(f"Saving -- buffer.size: {buffer.size}, manual size: {sum(map(lambda e: len(e), buffer.states))}, len(buffer.states): {len(buffer.states)}") # debug
        pass
    
    mse = float(torch.mean(torch.tensor(mses)))
    return actor_env_task, model_task, mse


if __name__ == '__main__':
    args = tyro.cli(Args)

    path_to_agaid = Path.cwd()
    assert path_to_agaid.name == "agaid"

    path_to_cleanrl = Path("../cleanrl").expanduser().resolve()
    assert path_to_cleanrl.is_dir()
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

        def forward(self, x, noisy=False):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            action = x * self.action_scale + self.action_bias
            if noisy: action += torch.normal(0, self.action_scale * 0.1)
            return action
    ###############################################################

    # Get actor_run_names from args.actor_run_name if provided else args.actor_run_name_pattern.
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
    # print(f"actor_run_names:", *actor_run_names, sep='\n')

    if not ray.is_initialized():
        ray.init()

    worker_ids = []
    ready_ids = []
    start_time = time.monotonic()
    for actor_run_name in actor_run_names:
        actor_path = path_to_cleanrl / f"runs/{args.env_id}/{actor_run_name}/td3_continuous_action.cleanrl_actor" / (f"_step_{args.actor_step}" if args.actor_step else '')
        # actor_path = path_to_cleanrl / f"runs/{args.env_id}/{actor_run_name}/td3_continuous_action.cleanrl_model" / (f"_step_{args.actor_step}" if args.actor_step else '')
        assert actor_path.is_file(), f"actor_path does not exist. actor_path:\n{actor_path}"

        # Seeding
        if args.seed:
            # random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Get actor_env_task.
        task_match = re.search('task_g_(?P<task>\d+(\.\d+)?)', actor_run_name)
        assert task_match is not None
        actor_env_task = float(task_match.group('task'))

        # assert args.env_id == 'Pendulum-v1' # env-specific task specification. Note: redo Actor for different policy algo.
        # env = gym.make(args.env_id, render_mode=None, g=task)
        env = gym.make('Pendulum-v1', render_mode=None, g=actor_env_task)
        actor = Actor(env)
        # actor.load_state_dict(torch.load(actor_path))
        save_dir = path_to_agaid / f"plot_logs/{args.env_id}/st_matrix/actor_env_task_{actor_env_task}"

        run_dir = path_to_agaid / 'runs' / args.env_id / args.exp_id

        buffer_path = path_to_agaid / f"offline_data/{args.env_id}/task_{actor_env_task}"
        buffer_path = list(buffer_path.iterdir())
        assert len(buffer_path) == 1
        buffer_path = buffer_path[0]
        buffer = MT_Model_Buffer.load(buffer_path, reinstantiate=True)
        buffer.compute_norms()
        buffer_husk = MT_Model_Buffer()
        buffer_husk.copy_norms(buffer)
        full_buffer = buffer
        buffer = buffer_husk
        del full_buffer

        # Set data_size and data_size_dir to the largest.
        max_data_size = 0
        max_data_size_dir = None
        for data_size_dir in run_dir.iterdir():
            data_size = int(data_size_dir.name[len('data_size_'):])
            if data_size > max_data_size:
                max_data_size = data_size
                max_data_size_dir = data_size_dir
        data_size = max_data_size
        data_size_dir = max_data_size_dir

        st_dir = data_size_dir / 'single_task'
        for task_dir in st_dir.iterdir():
            model_task = float(task_dir.name[len('task_'):])
            model_path = task_dir / 'aggregate' / 'aggregate.pkl'
            assert model_path.is_file()
            model = Dynamics_Model_Aggregate.load(model_path)

            save_path = save_dir / f"model_task_{model_task}/horizon_1_losses"

            assert len(worker_ids) <= args.num_workers
            if len(worker_ids) == args.num_workers:
                readied_ids, worker_ids = ray.wait(worker_ids, num_returns=1)
                ready_ids += readied_ids
            print(f"Beginning eval_st_cross_task for actor_env_task: {actor_env_task}, model_task: {model_task}")
            worker_id = ray.remote(eval_st_cross_task).remote(env=deepcopy(env), actor=ray.put(deepcopy(actor)), model=ray.put(model), actor_env_task=actor_env_task, model_task=model_task, buffer=ray.put(buffer), min_total_steps=args.min_total_steps, max_episode_steps=args.max_episode_steps, noisy_actions=args.noisy_actions, save_path=save_path)
            worker_ids.append(worker_id)
    st_cross_tasks = ray.get(ready_ids + worker_ids)
    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"Collection done. Duration: {duration/60:.2f} minutes")

    save_path = path_to_agaid / f"plot_logs/{args.env_id}/{args.exp_id}/st_cross_tasks.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as file:
        pickle.dump(st_cross_tasks, file)