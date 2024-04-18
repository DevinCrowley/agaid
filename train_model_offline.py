from pathlib import Path
from collections import defaultdict

# import ipdb as pdb #  for debugging
from ipdb import set_trace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter

from .mt_model_buffer import MT_Model_Buffer
from .dynamics_models import Dynamics_Model_Multihead, Dynamics_Model_Embed, Dynamics_Model_Aggregate


# batch_size=128#'input batch size for training (default: 64)'
# epochs=50#'number of epochs to train (default: 14)')
# lr=0.001#'learning rate (default: 1.0)')
# gamma=0.7#'Learning rate step gamma (default: 0.7)')
# no_cuda=False#'disables CUDA training')
# no_mps=False#'disables macOS GPU training')
# dry_run=False#'quickly check a single pass')
# log_interval=1000#'how many batches to wait before logging training status')
# save_model=True#'For Saving the current Model')



# obs_size = 10?
# pred_size = 6?
# num_tasks = 20?

buffer_file_path = 'some_path.pkl'

# Check if CUDA is available (i.e., GPU is available)
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU

logdir = Path('runs')


def train_model_offline(Env, model_type, buffer_file_path, logdir, lr, epochs, batch_size, test_size=0.2, device=None):
    obs_size = ?
    pred_size = ?
    

    logdir = Path(logdir)
    logdir.mkdir(parents=False, exist_ok=True)
    
    buffer = MT_Model_Buffer.read(buffer_file_path)
    writer = SummaryWriter(logdir)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_tasks = len(np.concatenate(*buffer.tasks).unique())
    if model_type == 'multihead':
        model = Dynamics_Model_Multihead(obs_size=obs_size, pred_size=pred_size, num_tasks=num_tasks)
        recurrent = False
    if model_type == 'embed':
        model = Dynamics_Model_Embed(obs_size=obs_size, pred_size=pred_size, num_tasks=num_tasks, embedding_dim=3)
        recurrent = False
    if model_type == 'aggregate':
        model = Dynamics_Model_Aggregate(obs_size=obs_size, pred_size=pred_size)
        recurrent = False
    else:
        raise RuntimeError(f"Unrecognized model_type: {model_type}")
    
    # Split buffer into train_buffer and test_buffer.
    train_buffer, test_buffer = buffer.train_test_split(test_size=test_size)

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


def log_train_test_info(writer, train_or_test, task_to_losses, avg_losses_across_tasks, epoch):
    assert train_or_test in ['train', 'test']

    # Write mean loss across all tasks and all state elements.
    writer.add_scalar(f"{train_or_test}/mean loss", avg_losses_across_tasks.mean(), epoch)
    # Write mean loss across all tasks for each state element.
    for state_idx in range(len(avg_losses_across_tasks)):
        writer.add_scalar(f"{train_or_test}/state element {state_idx}/mean loss", avg_losses_across_tasks[state_idx], epoch)
    for task, task_losses in task_to_losses.items():
        # Write mean loss for this task.
        writer.add_scalar(f"{train_or_test}/task losses/mean loss", task_losses.mean(), epoch)
        # Write avg loss for this task for each state element.
        for state_idx in range(len(task_losses)):
            writer.add_scalar(f"{train_or_test}/task losses/task_{task}/state element {state_idx}", task_losses[state_idx], epoch)


def train_model_offline_epoch(model, recurrent, buffer, batch_size, optimizer=None, device=None):
    """optimizer=None deactivates training. This is used for testing."""

    if optimizer is None:
        model.eval()
    else:
        model.train()

    for batch_idx, sample_batch in enumerate(buffer.sample(get_whole_trajectories=recurrent, batch_size=batch_size)):
        for e in sample_batch:
            e.to(device)
        task_batch, state_batch, action_batch, next_state_batch, done_batch, not_padding_mask = sample_batch

        # Normalize values.
        state_batch_normalized = buffer.normalize_state(state_batch, inplace=False)
        action_batch_normalized = buffer.normalize_action(action_batch, inplace=False)
        next_state_batch_normalized = buffer.normalize_state(next_state_batch, inplace=False)

        # Calculate residual.
        delta_normalized_state_batch = next_state_batch_normalized - state_batch_normalized # The learning target.

        # Get model prediction.
        observation_batch = torch.cat((state_batch_normalized, action_batch_normalized), dim=-1)
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
            avg_losses_across_tasks = losses.sum(dim=state_batch.shape()[:-1]) / not_padding_mask.sum(dim=state_batch.shape()[:-1]) # Shape: (state_size,).
            # task_batch shape: (batch_size, <max_traj_len if recurrent,>, <possibly task_size if > 1,>).
            task_to_losses = dict()
            for task in task_batch.unique():
                task_losses = losses[task_batch == task].mean(dim=0, dtype=float) # Shape: (state_size,).
                # Averaged across batch, one element per state element per task.
                task_to_losses[task] = task_losses
            return task_to_losses, avg_losses_across_tasks
