import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, trajectory=False):
        if not trajectory:
            # Sample a single transition
            return self.data[idx]
        else:
            # Sample a trajectory of transitions
            trajectory_length = 5  # Set the length of the trajectory
            if idx + trajectory_length > len(self.data):
                # If the trajectory exceeds the dataset length, adjust the start index
                start_idx = len(self.data) - trajectory_length
            else:
                start_idx = idx
            return self.data[start_idx:start_idx+trajectory_length]




# Define a collate function for DataLoader
def custom_collate_fn(batch, trajectory_sampling=False):
    if trajectory_sampling:
        return torch.stack(batch)
    else:
        return batch