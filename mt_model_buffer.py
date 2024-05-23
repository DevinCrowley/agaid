import itertools
from collections import namedtuple
from pathlib import Path
import pickle
from copy import deepcopy

# from ipdb import set_trace
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence


class MT_Model_Buffer:
    """
    Multi-task model learning buffer.
    This structured version stores experiences with a dimension for each trajectory rather than flattened.

    Assumes successive pushes always complete trajectories before calling other methods, 
    otherwise those incomplete trajectories may be cut off with self.end_episode.

    Stores:
        - tasks
        - states
        - actions
        - next_states
        - dones

    TODO: make into general superclass, subclassed for particular purposes, e.g. model vs policy training.
    """
    

    # def __len__(self):
    #     return len(self.states)


    # def __len__(self):
    #     """
    #     Return the number of experiences = the number of calls to self.push.
    #     """
    #     return self.size


    def __init__(self):
        # TODO: add optional max size?
        self.reset()


    def reset(self):
        """
        Clear memory.
        """

        # Accumulated by calls to self.push.
        # These are of the form [[trajectory_0: time_0, time_1,...], [trajectory_1: time_0, time_1, ...], ...].
        # These have ragged shape (number of trajectories, trajectory length).
        self.tasks = []
        self.states = []
        self.actions = []
        self.next_states = []
        self.dones = []

        # This is the number of stored experiences, i.e. the number of calls to self.push.
        self.size = 0

        # Prepare for calls to self.push.
        self._open_episode = False # This is True when an episode has been pushed to but hasn't been terminated or truncated.

    
    def shuffle_episodes(self, seed=None):
        """
        Shuffle episodes, i.e. trajectories.
        """
        self.end_episode()

        np.random.seed(seed=seed)
        shuffle_idxs = np.random.permutation(len(self.states))

        self.tasks = [self.tasks[idx] for idx in shuffle_idxs]
        self.states = [self.states[idx] for idx in shuffle_idxs]
        self.actions = [self.actions[idx] for idx in shuffle_idxs]
        self.next_states = [self.next_states[idx] for idx in shuffle_idxs]
        self.dones = [self.dones[idx] for idx in shuffle_idxs]

        return self


    def _start_episode(self):
        """
        Add new empty trajectory sequences.
        """
        self.tasks.append([])
        self.states.append([])
        self.actions.append([])
        self.next_states.append([])
        self.dones.append([])

        self._open_episode = True

    
    def end_episode(self):
        """
        This ends an episode, whether it be termination or truncation.
        To be called when a trajectory terminates -- with done=True in self.push -- or is truncated.
        Can be called after a terminated trajectory with no effect.
        """
        self._open_episode = False


    def push(self, task, state, action, next_state, done=False):
        """
        Stores information about this policy step.

        This implementation assumes a push with done=True will be called before other methods, 
        or else the incomplete trajectory may be truncated internally.

        Sets self._open_episode to True.

        Parameters
        ----------
        task : int/float/str/np.ndarray
            The task.
        state : np.ndarray
            The observed state.
        action : np.ndarray
            The action taken in that state.
        reward : float
            The instantaneous reward for this state and action.
        state : np.ndarray
            The next observed state.
        # done : bool
        #     Whether this action led to a terminal state.
        """
        if not self._open_episode:
            # Begin a new episode.
            self._start_episode()
        
        assert len(state) == len(next_state), f"len(state): {len(state)} doesn't match with len(next_state): {len(next_state)}"
        if len(self.tasks[-1]) > 0:
            assert task == self.tasks[-1][-1]

        # Cast to tensors and store.
        self.tasks[-1].append(torch.tensor(task, dtype=float))
        self.states[-1].append(torch.tensor(state, dtype=float))
        self.actions[-1].append(torch.tensor(action, dtype=float))
        self.next_states[-1].append(torch.tensor(next_state, dtype=float))
        self.dones[-1].append(torch.tensor(done, dtype=bool).reshape(1))

        self.size += 1

        if done:
            # This prepares for pushes as part of a new trajectory.
            self.end_episode()

    @staticmethod
    def _normalize(value, std, mean, inplace=False):
        
        if not inplace:
            if isinstance(value, torch.Tensor):
                value = torch.clone(value)
            elif isinstance(value, np.ndarray):
                value = np.copy(value)
            elif type(value) in [int, float]:
                pass
            else:
                raise RuntimeError(f"Unsupported type.\ntype(value): {type(value)}")
        
        if isinstance(value, torch.Tensor):
            value[...] = torch.div(torch.sub(value, mean), std)
        elif isinstance(value, np.ndarray):
            value[...] = (value - mean.detach().cpu().numpy()) / std.detach().cpu().numpy()
        else:
            value = (value - mean.detach().cpu().numpy()) / std.detach().cpu().numpy()
        return value
    def normalize_state(self, state, inplace=False):
        if not hasattr(self, '_state_std_mean'):
            self.compute_norms()
        std, mean = self._state_std_mean
        return self._normalize(value=state, std=std, mean=mean, inplace=inplace)
    def normalize_action(self, action, inplace=False):
        if not hasattr(self, '_action_std_mean'):
            self.compute_norms()
        std, mean = self._action_std_mean
        return self._normalize(value=action, std=std, mean=mean, inplace=inplace)
    def compute_norms(self):
        if not hasattr(self, '_state_std_mean'):
            # self._state_std_mean = torch.std_mean(torch.from_numpy(np.concatenate(self.states)), dim=0)
            self._state_std_mean = torch.std_mean(torch.cat(list(map(torch.stack, self.states))), dim=0)
        if not hasattr(self, '_action_std_mean'):
            # self._action_std_mean = torch.std_mean(torch.from_numpy(np.concatenate(self.actions)), dim=0)
            self._action_std_mean = torch.std_mean(torch.cat(list(map(torch.stack, self.actions))), dim=0)
    def copy_norms(self, buffer):
        if not hasattr(buffer, '_state_std_mean') or not hasattr(buffer, '_action_std_mean'):
            buffer.compute_norms()
        self._state_std_mean = buffer._state_std_mean
        self._action_std_mean = buffer._action_std_mean


    def sample(self, get_whole_trajectories, batch_size, drop_last=True):
        """
        Return a generator that yields randomly sampled mini-batches as (states, actions, returns, advantages, not_padding_mask) tuples.

        If get_whole_trajectories == False:
            Each element is a torch.Tensor with shape (batch_size,).
            states has shape (batch_size, <state_size>).

        elif get_whole_trajectories == True:
            Each element is a torch.Tensor with shape (batch_size, <longest_trajectory_length>).
            states has shape (batch_size, <longest_trajectory_length>, <state_size>).
            The <trajectory_length> dimension is padded to the length of the longest trajectory in the mini-batch.


        Parameters
        ----------
        get_whole_trajectories : bool
            If True, sample whole trajectories of experiences. If False, sample individual experiences.
            This should be True for recurrent policies.
        batch_size : int
            The number of experience tuples per mini-batch.
        """

        # Get the conversion from task values to task indices.
        task_idxs = self._get_task_idxs()
        
        if not get_whole_trajectories:
            # Sample individual experiences.
        
            # Flatten experiences and cast to torch.Tensor.
            flattened_tasks = torch.tensor(np.concatenate(task_idxs), dtype=float) # Retains the <task_size> dimension at the end.
            flattened_states = torch.tensor(np.concatenate(self.states), dtype=float) # Retains the <state_size> dimension at the end.
            flattened_actions = torch.tensor(np.concatenate(self.actions), dtype=float)
            flattened_next_states = torch.tensor(np.concatenate(self.next_states), dtype=float) # Retains the <state_size> dimension at the end.
            flattened_dones = torch.tensor(np.concatenate(self.dones), dtype=bool)
            # These went from shape (<number_of_trajectories>, <trajectory_length>, <state_size for states>)
            # to shape (<number_of_experiences>, <state_size for states>).

            # Get mini-batches of random indices of experience tuples.
            randomized_experience_indices = SubsetRandomSampler(range(len(flattened_states)))
            batch_indices_generator = BatchSampler(randomized_experience_indices, batch_size, drop_last=drop_last)

            # Yield mini-batches of experience tuples.
            for batch_indices in batch_indices_generator:
                # Select the experiences for this mini-batch.
                task_batch = flattened_tasks[batch_indices]
                state_batch = flattened_states[batch_indices]
                action_batch = flattened_actions[batch_indices]
                next_state_batch = flattened_next_states[batch_indices]
                done_batch = flattened_dones[batch_indices]

                # Get not_padding_mask: ones like the batch shape.
                not_padding_mask = torch.ones_like(done_batch) # There is no padding when get_whole_trajectories == False.

                yield task_batch.to(int), state_batch, action_batch, next_state_batch, done_batch, not_padding_mask
        
        else: # get_whole_trajectories == True
            # Sample whole trajectories of experiences.

            # Get mini-batches of random indices of trajectories.
            randomized_trajectory_indices = SubsetRandomSampler(range(len(self.states)))
            batch_indices_generator = BatchSampler(randomized_trajectory_indices, batch_size, drop_last=drop_last)

            # Yield mini-batches of whole trajectories.
            for batch_indices in batch_indices_generator:
                # Select the trajectories for this mini-batch.
                task_batch       = list(np.array(task_idxs       , dtype=object)[batch_indices])
                state_batch      = list(np.array(self.states     , dtype=object)[batch_indices])
                action_batch     = list(np.array(self.actions    , dtype=object)[batch_indices])
                next_state_batch = list(np.array(self.next_states, dtype=object)[batch_indices])
                done_batch       = list(np.array(self.done       , dtype=object)[batch_indices])
                
                # Get not_padding_mask: ones where the actual ragged trajectories come from real experiences.
                not_padding_mask = [np.ones_like(done_trajectory) for done_trajectory in done_batch]

                # Pad experience trajectories and cast to torch.Tensor.
                # TODO: upgrade this to use pack_padded_sequence.
                task_batch       = pad_sequence([torch.Tensor(task_trajectory      ) for task_trajectory       in task_batch      ], batch_first=True, padding_value=0.0)
                state_batch      = pad_sequence([torch.Tensor(state_trajectory     ) for state_trajectory      in state_batch     ], batch_first=True, padding_value=0.0)
                action_batch     = pad_sequence([torch.Tensor(action_trajectory    ) for action_trajectory     in action_batch    ], batch_first=True, padding_value=0.0)
                next_state_batch = pad_sequence([torch.Tensor(next_state_trajectory) for next_state_trajectory in next_state_batch], batch_first=True, padding_value=0.0)
                done_batch       = pad_sequence([torch.Tensor(done_trajectory      ) for done_trajectory       in done_batch      ], batch_first=True, padding_value=0.0)
                # Pad not_padding_mask to match the padded experiences.
                not_padding_mask = pad_sequence(not_padding_mask, batch_first=False, padding_value=0.0)
                # These went from shape (<batch_size>, <trajectory_length>, <state_size for states>)
                # to shape (<batch_size>, <longest_trajectory_length>, <state_size for states>).

                yield task_batch.to(int), state_batch, action_batch, next_state_batch, done_batch, not_padding_mask
    

    def _get_task_idxs(self):
        sorted_unique_tasks = np.sort(np.unique(np.concatenate(self.tasks)))
        sorted_unique_task_idxs = np.arange(len(sorted_unique_tasks))
        task_to_idx = dict(zip(sorted_unique_tasks, sorted_unique_task_idxs))
        task_idxs = np.empty(len(self.tasks), dtype=object)
        for i, task_traj in enumerate(self.tasks):
            unique_task = np.unique(task_traj).item() # Assumes each trajectory shares a task.
            task_idxs[i] = np.full_like(task_traj, task_to_idx[unique_task])
        task_idxs = list(task_idxs)
        return task_idxs



    def get_subset_buffer(self, num_eps=None, min_steps=None, no_extra=False, shuffle_episodes=True):
        assert (num_eps is None) ^ (min_steps is None), \
            f"Exactly one of 'num_eps' and 'min_steps' must be provided.\nnum_ep: {num_eps}, min_steps: {min_steps}"
        if self.size == 0:
            raise RuntimeError(f"get_subset_buffer should not be called on a buffer of size 0.")

        if shuffle_episodes: self.shuffle_episodes()
        subset_buffer = type(self)()

        if num_eps is None:
            # Determine number of trajectories to at least equal min_steps.
            size = 0
            for eps_idx in range(len(self.states)):
                size += len(self.states[eps_idx])
                if size >= min_steps:
                    # Enough steps sampled in these episodes.
                    break
            num_eps = eps_idx + 1

        subset_buffer.tasks = self.tasks[:num_eps]
        subset_buffer.states = self.states[:num_eps]
        subset_buffer.actions = self.actions[:num_eps]
        subset_buffer.next_states = self.next_states[:num_eps]
        subset_buffer.dones = self.dones[:num_eps]

        if min_steps is not None and no_extra:
            # Truncate last episode to exactly match min_steps.
            extra_steps = size - min_steps
            subset_buffer.tasks[-1] = subset_buffer.tasks[-1][:-(1+extra_steps)]
            subset_buffer.states[-1] = subset_buffer.states[-1][:-(1+extra_steps)]
            subset_buffer.actions[-1] = subset_buffer.actions[-1][:-(1+extra_steps)]
            subset_buffer.next_states[-1] = subset_buffer.next_states[-1][:-(1+extra_steps)]
            subset_buffer.dones[-1] = subset_buffer.dones[-1][:-(1+extra_steps)]

        subset_buffer.size = sum((len(state_traj) for state_traj in subset_buffer.states))
        return subset_buffer


    @classmethod
    def merge_buffers(cls, buffers):
        
        # Create new buffer object.
        merged_buffer = cls()

        for buffer in buffers:
            # End episode in incoming buffer.
            buffer.end_episode()
            # Copy pushed experiences.
            merged_buffer.tasks.extend(buffer.tasks)
            merged_buffer.states.extend(buffer.states)
            merged_buffer.actions.extend(buffer.actions)
            merged_buffer.next_states.extend(buffer.next_states)
            merged_buffer.dones.extend(buffer.dones)
            # Accumulate buffer sizes.
            merged_buffer.size += buffer.size
            
        return merged_buffer
    

    def train_test_split(self, test_size=None, train_size=None, split_by_episodes=False, shuffle=True, random_seed=None, view=True):
        """
        Following some conventions of sklearn.model_selection.train_test_split:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        test_size or train_size must be a float in [0, 1], indicating the proportion of elements in that partition.

        If split_by_episodes == True, then the split is performed on whole trajectories, irrespective of their sizes.
        If split_by_episodes == False, then the split is performed on whole trajectories to approximate the desired split on transitions.
        TODO: support option to split on transitions.
        """
        num_trajectories = len(self.states)
        assert num_trajectories >= 2, f"There must be at least 2 trajectories to do this method. num_trajectories: {num_trajectories}"
        if shuffle: self.shuffle_episodes(seed=random_seed)

        # Validate test_size and train_size to the proportion of steps in that partition.
        if not ((test_size is None) ^ (train_size is None)): # If both or neither is None:
            raise ValueError("Exactly one of test_size and train_size must be provided.")
        if train_size is not None:
            # Infer test_size from train_size.
            assert 0 <= train_size <= 1, f"train_size must be in [0, 1]. train_size: {train_size}"
            test_size = 1 - train_size
        else:
            # Infer train_size from test_size.
            assert 0 <= test_size <= 1, f"test_size must be in [0, 1]. test_size: {test_size}"
            train_size = 1 - test_size
        # train_size & test_size are in [0, 1].

        # Convert train_size & test_size to the appropriate number of episodes.
        if split_by_episodes:
            # Set train_size & test_size such that they represent the number of episodes requested.
            # Split whole trajectories such that train_size & test_size request the number of episodes.
            train_size = round(num_trajectories * train_size)
            test_size = num_trajectories - train_size
        else:
            # Set train_size & test_size such that they represent the number of episodes that balances the number of steps requested.
            # Convert sizes to the number of steps, rounding towards 50-50.
            train_size = round(self.size * train_size)
            test_size = self.size - train_size
            # Split whole trajectories such that train_size & test_size request the number of steps.
            if train_size < test_size:
                smaller_size = train_size
                smaller_size_is = 'train'
            else:
                smaller_size = test_size
                smaller_size_is = 'test'
            steps = 0
            # Get the number of episodes to get at least smaller_size steps.
            for ep_idx in range(len(self.states)):
                steps += len(self.states[ep_idx])
                if steps >= smaller_size:
                    # Enough steps selected.
                    break
            smaller_partition_num_eps = ep_idx + 1
            # Convert train_size & test_size to the number of episodes to select.
            if smaller_size_is == 'train':
                train_size = smaller_partition_num_eps
                test_size = num_trajectories - train_size
            elif smaller_size_is == 'test':
                test_size = smaller_partition_num_eps
                train_size = num_trajectories - test_size
        # train_size & test_size are numbers of trajectories.

        # Perform partition.
        # Instantiate empty buffers.
        train_buffer = self.__class__()
        test_buffer = self.__class__()
        # Partition and load data into buffers.
        for data_name in ['tasks', 'states', 'actions', 'next_states', 'dones']:
            buffer_data = getattr(self, data_name)
            # Partition data.
            train_buffer_data_partition = buffer_data[:train_size]
            test_buffer_data_partition = buffer_data[train_size:]
            # TODO: test whether view=True is broken upstream.
            if not view:
                train_buffer_data_partition = deepcopy(train_buffer_data_partition)
                test_buffer_data_partition = deepcopy(test_buffer_data_partition)
            # Load data.
            setattr(train_buffer, data_name, train_buffer_data_partition)
            setattr(test_buffer, data_name, test_buffer_data_partition)
        
        # Set buffer sizes.
        train_push_count = sum(map(lambda traj: len(traj), train_buffer.states))
        train_buffer.size = train_push_count
        test_push_count = sum(map(lambda traj: len(traj), test_buffer.states))
        test_buffer.size = test_push_count
        assert train_buffer.size > 0 and test_buffer.size > 0, f"train_buffer.size: {train_buffer.size}, test_buffer.size: {test_buffer.size}"

        return train_buffer, test_buffer
    

    def copy(self):
        return deepcopy(self)


    def save(self, file_path, create_dir=True, parents=True, overwrite_file=True):

        # TODO: dynamically generate unique file_name with hyperparameter hash.
        file_path = Path(file_path).with_suffix('.pkl')
        if create_dir:
            file_path.parent.mkdir(parents=parents, exist_ok=True)
        else:
            assert file_path.parent.is_dir(), f"create_dir={create_dir} and file_path is in non-existent directory.\nfile_path: {file_path}"
        if not overwrite_file:
            assert not file_path.is_file(), f"overwrite_file={overwrite_file} and file_path points to a file that already exists.\nfile_path: {file_path}"

        # Save self.
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    
    @classmethod
    def load(cls, file_path, reinstantiate=False):
        # Verify inputs.
        file_path = Path(file_path)
        if not file_path.is_file():
            raise RuntimeError(f"file_path {file_path} is not an existing file.")

        # Read data.
        with open(file_path, 'rb') as file:
            buffer = pickle.load(file)
        
        # Reinstantiate and copy attributes.
        if reinstantiate:
            new_buffer = cls()
            for attribute_name in ['tasks', 'states', 'actions', 'next_states', 'dones', 'size', '_open_episode']:
                attribute = getattr(buffer, attribute_name)
                setattr(new_buffer, attribute_name, attribute)
            buffer = new_buffer

        return buffer
