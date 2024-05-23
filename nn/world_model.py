from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def save(self, file_path, create_dir=True, overwrite_file=True):
        file_path = Path(file_path).with_suffix('.pkl')
        if create_dir:
            file_path.parent.mkdir(parents=False, exist_ok=True)
        else:
            assert file_path.parent.is_dir(), f"create_dir={create_dir} and file_path is in non-existent directory.\nfile_path: {file_path}"
        if not overwrite_file:
            assert not file_path.is_file(), f"overwrite_file={overwrite_file} and file_path points to a file that already exists.\nfile_path: {file_path}"

        #torch.save(self.state_dict(), file_path)
        pickle_target = {'args': self._args, 'kwargs': self._kwargs, 'state_dict': self.state_dict()}
        with open(file_path, 'wb') as file:
            pickle.dump(pickle_target, file)

    @classmethod
    def load(cls, file_path):
        file_path = Path(file_path).with_suffix('.pkl')
        if not file_path.is_file():
            raise RuntimeError(f"file_path {file_path} is not an existing file.")

        # Read data.
        with open(file_path, 'rb') as file:
            pickle_target = pickle.load(file)

        # Instantiate Net and load state_dict into it.
        try:
            net = cls(*pickle_target['args'], **pickle_target['kwargs'])
        except:
            net = cls(obs_size=4, pred_size=3)
        net.load_state_dict(pickle_target['state_dict'])
        
        net.eval()
        return net


class Dynamics_Model_Multihead(Net):
    """
    Tasks share the neural network features but have separate heads at the final layer.
    """
    
    def __init__(self, obs_size, pred_size, num_tasks=1):
        super().__init__(obs_size=obs_size, pred_size=pred_size, num_tasks=num_tasks)

        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.ModuleList([nn.Linear(512, pred_size) for _ in range(num_tasks)])

    def forward(self, x, task_indices):
        assert len(x) == len(task_indices)
        # batch_size = len(task_indices)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # task_indices has an element for each observation in the batch x.
        x = torch.stack([self.fc3[task_index](x[i]) for i, task_index in enumerate(task_indices)])
        return x


class Dynamics_Model_Embed(Net):
    """
    Tasks are embedded to a dense vector concatenated with the observation.
    """

    def __init__(self, obs_size, pred_size, num_tasks=1, embedding_dim=3):
        super().__init__(obs_size=obs_size, pred_size=pred_size, num_tasks=num_tasks, embedding_dim=embedding_dim)
        self.embedding = nn.Embedding(num_embeddings=num_tasks, embedding_dim=embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + obs_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, pred_size)

    def forward(self, x, task_indices):
        task_embeddings = self.embedding(task_indices)
        x = torch.cat((x, task_embeddings), axis=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Dynamics_Model_Aggregate(Net):
    """
    Single-task dynamics model. Treats multiple tasks as though a single task.
    """

    def __init__(self, obs_size, pred_size):
        super().__init__(obs_size=obs_size, pred_size=pred_size)
        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, pred_size)

    def forward(self, x, task_indices=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x          