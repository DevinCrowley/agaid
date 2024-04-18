import itertools
from collections import namedtuple
from pathlib import Path
import pickle
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence


class MT_Model_Buffer:
    """
    Multi-task model learning buffer.
    This structured version stores experiences with a dimension for each trajectory rather than flattened.

    Assumes successive pushes always complete trajectories before calling other methods, 
    otherwise those incomplete trajectories may be cut off with self._trim_buffer.

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
        # self._trimmed is True when self._trim_buffer has been called, unless self.reset or self.push or _start_trajectory have been called since.
        # self._trimmed = False # This indicates whether the latest trajectory is complete. This is set by _start_trajectory below.

        # Accumulated by calls to self.push.
        # These are of the form [[trajectory_0: time_0, time_1,...], [trajectory_1: time_0, time_1, ...], ...].
        # These have ragged shape (number of trajectories, trajectory length).
        self.tasks = []
        self.states = []
        self.actions = []
        self.next_states = []
        self.dones = []

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.welford_state_count = 1
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_var_sum = torch.ones(1) # mean2
        self.welford_action_count = 1
        self.welford_action_mean = torch.zeros(1)
        self.welford_action_var_sum = torch.ones(1) # mean2

        # This is the number of stored experiences, i.e. the number of calls to self.push.
        self.size = 0

        # Prepare for calls to self.push.
        self._start_trajectory()


    def _start_trajectory(self):
        """
        Add new empty trajectory sequences for states, actions, & rewards.
        """
        self.tasks.append([])
        self.states.append([])
        self.actions.append([])
        self.next_states.append([])
        self.dones.append([])

        self._trimmed = False

    
    def end_trajectory(self):
        """
        To be called to truncate a trajectory that hasn't itself terminated with done=True in self.push.
        Can be called after a terminated trajectory with no effect.
        Calls _start_trajectory if it hasn't just been called.
        """
        if not self._trimmed:
            self._start_trajectory()


    def push(self, task, state, action, next_state, done=False):
        """
        Stores information about this policy step.

        This implementation assumes a push with done=True will be called before other methods, 
        or else the incomplete trajectory may be trimmed and discarded.

        Sets self._trimmed to False.

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
        # self._trim_buffer must be called after the last call to self.push before calling self.sample.
        if self._trimmed: self._start_trajectory() # Allow pushing to resume after sampling.
        assert not self._trimmed, "Call to self.push after self._trim_buffer without call to self._start_trajectory."
        assert len(state) == len(next_state), f"len(state): {len(state)} doesn't match with len(next_state): {len(next_state)}"
        
        if len(self.tasks[-1]) > 0:
            assert task == self.tasks[-1][-1]

        self.tasks[-1].append(task)
        self.states[-1].append(state)
        self.actions[-1].append(action)
        self.next_states[-1].append(next_state)
        self.dones[-1].append(done)

        self.size += 1

        if done:
            # This prepares for pushes as part of a new trajectory.
            self._start_trajectory()

        # Update welford normalization stats.
        self.welford_state_count += 1
        self.welford_action_count += 1

        # state
        delta = state - self.welford_state_mean
        self.welford_state_mean += delta / self.welford_state_count
        delta2 = state - self.welford_state_mean
        self.welford_state_var_sum += delta * delta2
        
        # action
        delta = action - self.welford_action_mean
        self.welford_action_mean += delta / self.welford_action_count
        delta2 = action - self.welford_action_mean
        self.welford_action_var_sum += delta * delta2
    

    def _trim_buffer(self):
        """
        Prepare for a call to self.sample by normalizing advantages and trimming empty sequences from self._start_trajectory.
        Cannot call self.push after self._trim_buffer until self.reset.

        Must be called after ending the last trajectory.
        """
        # Assert buffer is nonempty.
        assert len(self.states) > 0, "No states have been pushed yet."
        # # Assert the latest trajectory has been ended.
        # assert len(self.returns) == len(self.states), "self.end_trajectory has not been called on the latest trajectory."
        # Assert that self._start_trajectory has been called since the last push.
        # assert len(self.states[-1]) == len(self.actions[-1]) == len(self.rewards[-1]) == len(self.dones[-1]) == 0, \
        #     "self._start_trajectory has not been called since the last push."
        
        # Trim empty sequences from self._start_trajectory. OR incomplete sequences.
        del self.tasks[-1]
        del self.states[-1]
        del self.actions[-1]
        del self.next_states[-1]
        del self.dones[-1]

        self._trimmed = True


    def sample(self, get_whole_trajectories, batch_size):
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
        if not self._trimmed:
            # self._trim_buffer sets self._trimmed to True.
            self._trim_buffer()
        assert self._trimmed
        
        if not get_whole_trajectories:
            # Sample individual experiences.
        
            # Flatten experiences and cast to torch.Tensor.
            flattened_tasks = torch.Tensor(np.concatenate(self.tasks)) # Retains the <task_size> dimension at the end.
            flattened_states = torch.Tensor(np.concatenate(self.states)) # Retains the <state_size> dimension at the end.
            flattened_actions = torch.Tensor(np.concatenate(self.actions))
            flattened_next_states = torch.Tensor(np.concatenate(self.next_states)) # Retains the <state_size> dimension at the end.
            flattened_dones = torch.Tensor(np.concatenate(self.dones))
            # These went from shape (<number_of_trajectories>, <trajectory_length>, <state_size for states>)
            # to shape (<number_of_experiences>, <state_size for states>).

            # Get mini-batches of random indices of experience tuples.
            randomized_experience_indices = SubsetRandomSampler(range(len(flattened_states)))
            batch_indices_generator = BatchSampler(randomized_experience_indices, batch_size, drop_last=True)

            # Yield mini-batches of experience tuples.
            for batch_indices in batch_indices_generator:
                # Select the experiences for this mini-batch.
                task_batch = flattened_tasks[batch_indices]
                state_batch = flattened_states[batch_indices]
                action_batch = flattened_actions[batch_indices]
                next_state_batch = flattened_next_states[batch_indices]
                done_batch = flattened_dones[batch_indices]

                not_padding_mask = 1 # There is no padding when get_whole_trajectories == False.

                yield task_batch, state_batch, action_batch, next_state_batch, done_batch, not_padding_mask
        
        else: # get_whole_trajectories == True
            # Sample whole trajectories of experiences.

            # Get mini-batches of random indices of trajectories.
            randomized_trajectory_indices = SubsetRandomSampler(range(len(self.states)))
            batch_indices_generator = BatchSampler(randomized_trajectory_indices, batch_size, drop_last=True)

            # Yield mini-batches of whole trajectories.
            for batch_indices in batch_indices_generator:
                # Select the trajectories for this mini-batch.
                task_batch       = list(np.array(self.tasks      , dtype=object)[batch_indices])
                state_batch      = list(np.array(self.states     , dtype=object)[batch_indices])
                action_batch     = list(np.array(self.actions    , dtype=object)[batch_indices])
                next_state_batch = list(np.array(self.next_states, dtype=object)[batch_indices])
                done_batch       = list(np.array(self.done       , dtype=object)[batch_indices])
                
                # Get not_padding_mask: ones where the actual ragged trajectories come from real experiences.
                not_padding_mask = [np.ones_like(returns_trajectory) for returns_trajectory in done_batch]

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

                yield task_batch, state_batch, action_batch, next_state_batch, done_batch, not_padding_mask

    
    @staticmethod
    def _normalize(value, welford_count, welford_mean, welford_var_sum, inplace=False):
        variance = welford_var_sum / welford_count
        stdev = np.sqrt(variance)
        
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
            value[...] = torch.div(torch.sub(value, welford_mean), stdev)
        else:
            value[...] = (value - welford_mean) / stdev
        return value
    def normalize_state(self, state, inplace=False):
        return self._normalize(value=state, welford_count=self.welford_state_count, welford_mean=self.welford_state_mean, welford_var_sum=self.welford_state_var_sum, inplace=inplace)
    def normalize_action(self, action, inplace=False):
        return self._normalize(value=action, welford_count=self.welford_action_count, welford_mean=self.welford_action_mean, welford_var_sum=self.welford_action_var_sum, inplace=inplace)

    
    @staticmethod
    def _denormalize(value, welford_count, welford_mean, welford_var_sum, inplace=False):
        variance = welford_var_sum / welford_count
        stdev = torch.sqrt(variance)
        
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
            value[...] = torch.mul(value, stdev).add(welford_mean)
        else:
            value[...] = value * stdev + welford_mean
        return value
    def denormalize_state(self, state, inplace=False):
        return self._denormalize(value=state, welford_count=self.welford_state_count, welford_mean=self.welford_state_mean, welford_var_sum=self.welford_state_var_sum, inplace=inplace)
    def denormalize_action(self, action, inplace=False):
        return self._denormalize(value=action, welford_count=self.welford_action_count, welford_mean=self.welford_action_mean, welford_var_sum=self.welford_action_var_sum, inplace=inplace)
    
    
    @staticmethod
    def _denormalize(value, welford_state_count, welford_mean, welford_var_sum, inplace=False):
        variance = welford_var_sum / welford_state_count
        stdev = np.sqrt(variance)
        
        if not inplace:
            if isinstance(value, np.ndarray):
                value = np.copy(value)
            elif isinstance(value, torch.Tensor):
                value = torch.clone(value)
            elif type(value) in [int, float]:
                pass
            else:
                raise RuntimeError(f"Unsupported type.\ntype(value): {type(value)}")
        
        value[...] = value * stdev + welford_mean
        return value
    def denormalize_state(self, state, inplace=False):
        return self._denormalize(value=state, welford_state_count=self.size, welford_mean=self.welford_state_mean, welford_var_sum=self.welford_state_var_sum, inplace=inplace)
    def denormalize_action(self, action, inplace=False):
        return self._denormalize(value=action, welford_state_count=self.size, welford_mean=self.welford_action_mean, welford_var_sum=self.welford_action_var_sum, inplace=inplace)


    @classmethod
    def merge_buffers(cls, buffers):
        
        # Assert buffers have not tried to sample yet.
        for buffer in buffers:
            assert not buffer._trimmed, "buffer._trimmed should be False for all buffers, \
                else they may have already attempted to sample and have called _trim_buffer. \
                This is a problem because we want to call _trim_buffer to normalize advantages after merge_buffers."
        
        # Create new buffer object.
        merged_buffer = cls()
        
        # Trim empty sequences from self._start_trajectory.
        # del merged_buffer.tasks[-1]
        # del merged_buffer.states[-1]
        # del merged_buffer.actions[-1]
        # del merged_buffer.next_states[-1]
        # del merged_buffer.dones[-1]
        merged_buffer._trim_buffer()

        for buffer in buffers:
            # Trim incoming buffer.
            buffer._trim_buffer()
            # Copy pushed experiences.
            merged_buffer.tasks.extend(buffer.tasks)
            merged_buffer.states.extend(buffer.states)
            merged_buffer.actions.extend(buffer.actions)
            merged_buffer.next_states.extend(buffer.next_states)
            merged_buffer.dones.extend(buffer.dones)
            # Accumulate buffer sizes.
            merged_buffer.size += buffer.size
        
        # Prepare merged_buffer for possible future pushes.
        merged_buffer._start_trajectory()
        
        # TODO: merge buffer welford normalization stats.
        raise NotImplementedError('TODO: implement merger of welford normalization statistics.')
            
        return merged_buffer
    

    def train_test_split(self, test_size=None, train_size=None, random_state=None, shuffle=True, view=True):
        """
        Following the conventions of sklearn.model_selection.train_test_split:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        test_size or train_size can be an int, indicating the number of elements in that partition, 
        or a float in [0, 1], indicating the proportion of elements in that partition.

        Elements are whole trajectories.
        """
        self._trim_buffer()
        num_trajectories = len(self.states)

        # Validate test_size and train_size to the number of trajectories in that partition.
        if (test_size is None) ^ (train_size is None): # If both or neither is None:
            raise ValueError("Exactly one of test_size and train_size must be provided.")
        # Infer test_size from train_size.
        if train_size is not None:
            if train_size > 1:
                test_size = num_trajectories - train_size
            else:
                test_size = 1 - train_size
        # Validate test_size.
        if test_size > 1:
            # Interpret as number of trajectories in test partition.
            test_size = int(test_size)
            assert test_size <= num_trajectories
        else:
            assert 0 <= test_size <= 1
            test_size = int(num_trajectories * test_size)
        # Infer train_size from validated test_size.
        train_size = num_trajectories - test_size

        # Perform partition.
        test_train_split_buffers = [None, None]
        for partition_idx, partition_size in enumerate([test_size, train_size]):
            # Instantiate empty buffer.
            test_train_split_buffers[partition_idx] = self.__class__()
            test_train_split_buffers[partition_idx]._trim_buffer()
            # Partition and load data into buffers.
            for traj_data_name in ['tasks', 'states', 'actions', 'next_states', 'dones']:
                traj_data = getattr(self, traj_data_name)
                if shuffle:
                    # Shuffle in place.
                    np.random.seed(random_state)
                    np.random.shuffle(traj_data)
            # Partition data.
            traj_data_partition = traj_data[:partition_size]
            # TODO: test whether view=True is broken upstream.
            if not view: traj_data_partition = deepcopy(traj_data_partition)
            # Load data.
            setattr(test_train_split_buffers[partition_idx], traj_data_name, traj_data_partition)

            # Restore buffer to a push-ready state.
            test_train_split_buffers[partition_idx]._start_trajectory()
            push_count = np.sum(list(map(lambda traj: len(traj), test_train_split_buffers[partition_idx].states)))
            test_train_split_buffers[partition_idx].size = push_count

        return *test_train_split_buffers, # test_buffer, train_buffer
            
    

    def save(self, file_path, create_dir=True, overwrite_file=True):

        # TODO: dynamically generate unique file_name with hyperparameter hash.
        file_path = Path(file_path)
        if create_dir:
            file_path.parent.mkdir(parents=False, exist_ok=True)
        else:
            assert file_path.parent.is_dir(), f"create_dir={create_dir} and file_path is in non-existent directory.\nfile_path: {file_path}"
        if not overwrite_file:
            assert not file_path.is_file(), f"overwrite_file={overwrite_file} and file_path points to a file that already exists.\nfile_path: {file_path}"
        

        # Save self.
        with open(file_path, 'wb') as file:
            pickle.dump(file, self)

    
    @classmethod
    def load(cls, file_path):
        # Verify inputs.
        file_path = Path(file_path)
        if not file_path.is_file():
            raise RuntimeError(f"file_path {file_path} is not an existing file.")

        # Read data.
        with open(file_path, 'rb') as file:
            return pickle.load(file)
