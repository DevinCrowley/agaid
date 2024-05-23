import random
import re
import time
from pathlib import Path
from collections import defaultdict
from importlib import  import_module

# import ipdb as pdb #  for debugging
from ipdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import tyro
from dataclasses import dataclass
from copy import deepcopy
import gymnasium as gym
import ray

from mt_model_buffer import MT_Model_Buffer
from nn.world_model import Dynamics_Model_Multihead, Dynamics_Model_Embed, Dynamics_Model_Aggregate


@dataclass
class Args:
    env_id: str = "Pendulum-v1"
    """The id of the environment"""
    exp_id: str = "200k_0"
    """The id of the experiment"""
    batch_size: int = 128
    """The number of steps per mini-batch"""
    epochs: int = 50
    """The number of epochs of training"""
    test_size: float = 0.1
    """The proportion of data kept out for testing"""
    lr: float = 1e-3
    """The number of epochs of training"""
    buffer_name_pattern: str = "min_total_steps_100000__actor__[\w-]+\.pkl"
    # buffer_name_pattern: str = "min_total_steps_100000__actor__Pendulum-v1__td3_continuous_action__task_g_0.0_1__1713749400"
    """The regex pattern for the unique buffer to load per task"""
    num_workers: int = 1
    """the number of ray workers"""
    # save_interval: int = 1000
    # """The number of epochs of training"""
    # task_distribution: int = 1000
    # """The number of epochs of training"""
    # cuda: bool = True
    # """if toggled, cuda will be enabled by default"""
    
    # num_envs: int = 1
    # """the number of parallel game environments"""
    # buffer_size: int = int(1e6)
    # """the replay memory buffer size"""


def train_model_offline(env_id, exp_id, data_size, task, model_type, train_buffer, test_buffer, lr, epochs, batch_size, device=None):
    
    # Get obs_size and pred_size from env.
    env = gym.make(env_id, render_mode=None)
    env_obs_size = np.prod(env.observation_space.shape)
    env_act_size = np.prod(env.action_space.shape)
    assert len(env.observation_space.shape) == 1 == len(env.action_space.shape)
    obs_size = env_obs_size + env_act_size
    pred_size = env_obs_size
    
    logdir = Path('runs')
    assert logdir.is_dir()
    logdir /= env_id
    logdir /= exp_id
    logdir /= f"data_size_{data_size}"
    if isinstance(task, str):
        logdir /= f"multi_task"
    else:
        assert isinstance(task, float)
        logdir /= f"single_task"
    logdir /= f"task_{task}"
    logdir /= model_type
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    num_tasks = len(np.unique(np.concatenate([*train_buffer.tasks, *test_buffer.tasks])))
    if model_type == 'multihead':
        model = Dynamics_Model_Multihead(obs_size=obs_size, pred_size=pred_size, num_tasks=num_tasks)
        recurrent = False
    elif model_type == 'embed':
        model = Dynamics_Model_Embed(obs_size=obs_size, pred_size=pred_size, num_tasks=num_tasks, embedding_dim=3)
        recurrent = False
    elif model_type == 'aggregate':
        model = Dynamics_Model_Aggregate(obs_size=obs_size, pred_size=pred_size)
        recurrent = False
    else:
        raise RuntimeError(f"Unrecognized model_type: {model_type}")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model, write logging info.
    best_avg_loss = torch.inf
    for epoch in range(epochs):
        # Train model for one epoch.
        task_to_losses, avg_losses_across_tasks = train_model_offline_epoch(model=model, recurrent=recurrent, buffer=train_buffer, batch_size=batch_size, optimizer=optimizer, device=device)
        
        # Save updated model.
        if avg_losses_across_tasks.mean() < best_avg_loss:
            model.save(logdir / model_type)
        
        # Log training info.
        log_train_test_info(writer=writer, train_or_test='train', task_to_losses=task_to_losses, avg_losses_across_tasks=avg_losses_across_tasks, epoch=epoch)

        # Test model for one epoch.
        task_to_losses, avg_losses_across_tasks = train_model_offline_epoch(model=model, recurrent=recurrent, buffer=test_buffer, batch_size=batch_size, optimizer=None, device=device)
        
        # Log testing info.
        log_train_test_info(writer=writer, train_or_test='test', task_to_losses=task_to_losses, avg_losses_across_tasks=avg_losses_across_tasks, epoch=epoch)

    print(f"Completed training model:\ndata_size: {data_size}\ntask: {task}\nmodel_type: {model_type}")
    return


def log_train_test_info(writer, train_or_test, task_to_losses, avg_losses_across_tasks, epoch):
    assert train_or_test in ['train', 'test']

    # Write mean loss across all tasks and all state elements.
    writer.add_scalar(f"{train_or_test}/mean loss", avg_losses_across_tasks.mean(), epoch)
    # Write mean loss across all tasks for each state element.
    for state_idx in range(len(avg_losses_across_tasks)):
        # writer.add_scalar(f"{train_or_test}/state element {state_idx}/mean loss", avg_losses_across_tasks[state_idx], epoch)
        writer.add_scalar(f"{train_or_test}/state_loss_across_tasks/state_element_{state_idx}", avg_losses_across_tasks[state_idx], epoch)
    for task, task_losses in task_to_losses.items():
        # Write mean loss for this task.
        # TODO: check this one, it had 6x entries per epoch?
        writer.add_scalar(f"{train_or_test}/task_losses/mean_loss", task_losses.mean(), epoch)
        # Write avg loss for this task for each state element.
        for state_idx in range(len(task_losses)):
            writer.add_scalar(f"{train_or_test}/task_losses/task_{task}/state_element_{state_idx}", task_losses[state_idx], epoch)


def train_model_offline_epoch(model, recurrent, buffer, batch_size, optimizer=None, device=None):
    """optimizer=None deactivates training. This is used for testing."""

    if optimizer is None:
        model.eval()
    else:
        model.train()

    epoch_task_to_losses = []
    epoch_avg_losses_across_tasks = []
    all_unique_tasks = set()
    for batch_idx, sample_batch in enumerate(buffer.sample(get_whole_trajectories=recurrent, batch_size=batch_size, drop_last=False)):
        for e in sample_batch:
            e.to(device)
        task_batch, state_batch, action_batch, next_state_batch, done_batch, not_padding_mask = sample_batch
        all_unique_tasks.update(np.unique(task_batch))

        # Normalize values.
        state_batch_normalized = buffer.normalize_state(state_batch, inplace=False)
        action_batch_normalized = buffer.normalize_action(action_batch, inplace=False)
        next_state_batch_normalized = buffer.normalize_state(next_state_batch, inplace=False)

        # Calculate residual.
        delta_normalized_state_batch = next_state_batch_normalized - state_batch_normalized # The learning target.

        # Get model prediction.
        observation_batch = torch.cat((state_batch_normalized, action_batch_normalized), dim=-1).to(torch.float)
        predicted_normalized_residual_batch = model(observation_batch, task_batch) # prediction is predicted normalized next_state residual.

        # Compute loss.
        if optimizer is not None: optimizer.zero_grad()
        # loss = F.mse_loss(input=predicted_normalized_residual_batch, target=delta_normalized_state_batch) # Doesn't account for not_padding_mask.
        losses = ((predicted_normalized_residual_batch - delta_normalized_state_batch) * not_padding_mask).pow(2) # Shape: (batch_size, <max_traj_len if recurrent,> state_size).
        loss = losses.sum() / not_padding_mask.sum()
        if optimizer is not None:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01) #max_norm=grad_clip=?0.01
            optimizer.step()

        with torch.no_grad():
            predicted_normalized_next_state_batch = predicted_normalized_residual_batch + state_batch_normalized
            # prediction_denormalized_next_state_batch = buffer.denormalize_state(predicted_normalized_next_state_batch)
            # normalized_mses = F.mse_loss(predicted_normalized_next_state_batch, next_state_batch_normalized, reduction='none')
            avg_losses_across_tasks = losses.sum(dim=list(range(state_batch.dim() - 1))) / not_padding_mask.sum(dim=list(range(state_batch.dim() - 1))) # Shape: (state_size,).
            # task_batch shape: (batch_size, <max_traj_len if recurrent,>, <possibly task_size if > 1,>).
            task_to_losses = dict()
            for task in np.unique(task_batch):
                # task_losses = losses[task_batch == task].mean(dim=0, dtype=float) # Shape: (state_size,). # NOTE: doesn't account for not_padding_mask.
                try:
                    task_losses = losses[task_batch == task].sum(dim=0, dtype=float) / not_padding_mask[task_batch == task].sum() # Shape: (state_size,).
                except:
                    print(f"DEBUG PRINTOUTS") # debug
                    print(f"losses.shape:               {losses.shape}") # debug
                    print(f"task_batch.shape:           {task_batch.shape}") # debug
                    print(f"not_padding_mask.shape:     {not_padding_mask.shape}") # debug
                    print(f"(task_batch == task).shape: {(task_batch == task).shape}") # debug
                    print(f"np.unique(task_batch): {np.unique(task_batch)}") # debug
                    raise
                # Averaged across batch, one element per state element per task.
                task_to_losses[task] = task_losses
            epoch_task_to_losses.append({key: value.detach().to('cpu').numpy() for key, value in task_to_losses.items()})
            epoch_avg_losses_across_tasks.append(avg_losses_across_tasks.detach().to('cpu').numpy())

    epoch_task_to_losses = {task: np.mean(list(task_to_losses[task] for task_to_losses in epoch_task_to_losses if task in task_to_losses), axis=0) for task in all_unique_tasks}
    epoch_avg_losses_across_tasks = np.mean(epoch_avg_losses_across_tasks, axis=0)
    return epoch_task_to_losses, epoch_avg_losses_across_tasks


if __name__ == "__main__":
    overall_start = time.monotonic()
    args = tyro.cli(Args)

    # Check if CUDA is available (i.e., GPU is available)
    if torch.cuda.is_available(): device = torch.device("cuda") # Use GPU
    else: device = torch.device("cpu") # Use CPU

    if not ray.is_initialized():
        ray.init()
    
    path_to_agaid = Path.cwd()
    assert path_to_agaid.name == "agaid"
    buffer_tasks_dir = path_to_agaid / f"offline_data/{args.env_id}"
    assert buffer_tasks_dir.is_dir(), f"buffer_tasks_dir does not exist.\nbuffer_tasks_dir:\n {buffer_tasks_dir}"
    # agaid / offline_data / env_id / task_val / buffers

    pattern = re.compile(args.buffer_name_pattern)
    task_to_buffer_paths = dict() # One name for each task
    tasks = []
    for buffer_dir in buffer_tasks_dir.iterdir():
        task = float(buffer_dir.name[len('task_'):])
        tasks.append(task)
        buffer_paths = []
        for buffer_path in buffer_dir.iterdir():
            if pattern.fullmatch(buffer_path.name): buffer_paths.append(buffer_path)
        assert len(buffer_paths) == 1, f"buffer_name_pattern must match exactly 1 buffer file per task subfolder "\
            f"but instead matched {len(buffer_paths)}.\nbuffer_dir: {buffer_dir}\nbuffer_paths: {buffer_paths}"
        buffer_path = buffer_paths[0]
        task_to_buffer_paths[task] = buffer_path
    # print(f"buffer_names:", *(path.name for path in task_buffer_paths), sep='\n') # debug

    data_sizes = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, int(100000*(1-args.test_size))]).astype(int) # In ascending order of size.
    # data_sizes = np.append(100 * 2**np.arange(10), 90000).astype(int) # In ascending order of size.
    # data_sizes = np.geomspace(100, 100000, 10).astype(int) # In ascending order of size.
    # data_sizes = np.flip(np.sort(np.geomspace(5000, 100, 2).astype(int))) # In descending order of size. #  TODO:  -------------------------->RESTORE   TODO

    # Load buffers for all tasks.
    task_to_train_buffer = dict()
    task_to_test_buffer = dict()
    start_time = time.monotonic()
    print(f"Loading {len(task_to_buffer_paths)} buffers")
    for task, buffer_path in task_to_buffer_paths.items(): #  TODO:  -------------------------->RESTORE   TODO
        buffer = MT_Model_Buffer.load(buffer_path, reinstantiate=True)
        train_buffer, test_buffer = buffer.train_test_split(test_size=args.test_size)
        task_to_train_buffer[task] = train_buffer
        task_to_test_buffer[task] = test_buffer
    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"Buffers loaded in {duration:.2f} seconds")

    worker_ids = []
    ready_ids = []
    for data_size in data_sizes:
        print(f"Sampling and merging buffers for data_size {data_size}")
        # Note: these methods may share buffer memory.
        # Subsample task buffers.
        task_to_sub_train_buffer = {task: train_buffer.get_subset_buffer(min_steps=data_size, no_extra=True) for task, train_buffer in task_to_train_buffer.items()}
        task_to_sub_test_buffer = {task: test_buffer.get_subset_buffer(min_steps=data_size, no_extra=True) for task, test_buffer in task_to_test_buffer.items()}
        # Combine task buffers.
        mt_buffer_train = MT_Model_Buffer.merge_buffers(task_to_sub_train_buffer.values()).shuffle_episodes()
        mt_buffer_test = MT_Model_Buffer.merge_buffers(task_to_sub_test_buffer.values()).shuffle_episodes()

        # # Train single_task models and a multi_task model for each model_type.
        # for model_type in ['multihead', 'embed', 'aggregate']:
        #     print(f"Training models for model_type {model_type}")
        #     # Train the multi_task model 'all' and the single_task models.
        #     for task in ['all', *task_to_sub_train_buffer.keys()]:
        #         print(f"Training model for task {task}")
        #         # Set train_buffer and test_buffer.
        #         if task == 'all':
        #             train_buffer_id = ray.put(mt_buffer_train.copy())
        #             test_buffer_id = ray.put(mt_buffer_test.copy())
        #         else:
        #             assert isinstance(task, float), f"task is expected to be a float for single-task models.\ntype(task): {type(task)}\ntask: {task}"
        #             train_buffer_id = ray.put(task_to_sub_train_buffer[task].copy())
        #             test_buffer_id = ray.put(task_to_sub_test_buffer[task].copy())
        #         # Assign ray workers.
        #         assert len(worker_ids) <= args.num_workers
        #         if len(worker_ids) == args.num_workers:
        #             readied_ids, worker_ids = ray.wait(worker_ids, num_returns=1)
        #             ready_ids += readied_ids
        #         try:
        #             worker_id = ray.remote(train_model_offline).remote(env_id=args.env_id, exp_id=args.exp_id, data_size=data_size, task=task, model_type=model_type, \
        #                                 train_buffer=train_buffer_id, test_buffer=test_buffer_id, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=device)
        #         except:
        #             print(f"ERROR: train_buffer.size: {train_buffer.size}, test_buffer.size: {test_buffer.size}")
        #             raise
        #         worker_ids.append(worker_id)

        st_model_types = ['aggregate']
        mt_model_types = ['multihead', 'embed', 'aggregate']
        # Train the multi_task model 'all' and the single_task models.
        for task in ['all'] + tasks: # 'all' is the multi-task case, tasks are the individual tasks for the single-task case.
            # Set train_buffer and test_buffer.
            print(f"Training model for task {task}")
            if isinstance(task, str):
                # Train multi-task models.
                assert task == 'all'
                train_buffer_id = ray.put(mt_buffer_train.copy())
                test_buffer_id = ray.put(mt_buffer_test.copy())
                model_types = mt_model_types
            else:
                # Train single-task models.
                assert isinstance(task, float), f"task is expected to be a float for single-task models.\ntype(task): {type(task)}\ntask: {task}"
                train_buffer_id = ray.put(task_to_sub_train_buffer[task].copy())
                test_buffer_id = ray.put(task_to_sub_test_buffer[task].copy())
                model_types = st_model_types
            for model_type in model_types:
                # Assign ray workers.
                assert len(worker_ids) <= args.num_workers
                if len(worker_ids) == args.num_workers:
                    readied_ids, worker_ids = ray.wait(worker_ids, num_returns=1)
                    ready_ids += readied_ids
                try:
                    worker_id = ray.remote(train_model_offline).remote(env_id=args.env_id, exp_id=args.exp_id, data_size=data_size, task=task, model_type=model_type, \
                                        train_buffer=train_buffer_id, test_buffer=test_buffer_id, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=device)
                except:
                    print(f"ERROR: train_buffer.size: {train_buffer.size}, test_buffer.size: {test_buffer.size}")
                    raise
                worker_ids.append(worker_id)
    ray.get(ready_ids + worker_ids)
    overall_end = time.monotonic()
    overall_duration = overall_end - overall_start
    print(f"Overall duration: {overall_duration/60:.2f} minutes")