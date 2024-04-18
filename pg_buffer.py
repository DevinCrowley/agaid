from abc import ABC, abstractmethod
from .abc_buffer import ABC_Buffer
import itertools
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence


class PG_Buffer(ABC_Buffer):
    """
    Policy Gradient Buffer: this is the basic buffer class for policy gradient algorithms.
    This structured version stores experiences with a dimension for each trajectory rather than flattened.

    Methods:
        - reset
            Clear memory.
        - _start_trajectory
            Add empty sequences to self.states, self.actions, self.rewards, & self.values.
        - push
            Store information at this policy step. 
            These are appended to the last elements of self.states, self.actions, self.rewards, & self.values.
        - end_trajectory
            Compute returns and call _start_trajectory.
        - _finish_buffer
            Compute advantages and trim empty sequences created by _start_trajectory.
        - sample
            Yield mini-batches of states, actions, returns, & advantages.
    """
    

    # def __len__(self):
    #     return len(self.states)


    def __init__(self, discount_factor_gamma, GAE_lambda=1):
        """
        _summary_

        Parameters
        ----------
        discount_factor_gamma : float
            The time discount factor (gamma) for discounting future rewards.
        GAE_lambda : float
            The exponential weight discount used in the GAE advantage estimate.
            0 results in high bias, 1 results in high variance. Must be in [0, 1].
        """
        # discount_factor_gamma = gamma.
        discount_factor_gamma = float(discount_factor_gamma)
        assert 0 <= discount_factor_gamma <= 1
        self.discount_factor_gamma = discount_factor_gamma

        GAE_lambda = float(GAE_lambda)
        assert 0 <= GAE_lambda <= 1, "GAE_lambda must be in [0, 1]."
        self.GAE_lambda = GAE_lambda

        self.reset()


    def reset(self):
        """
        Clear memory.
        """
        # self.ready_to_sample is True when self._finish_buffer has been called, unless self.reset or self.push has been called since.
        self.ready_to_sample = False

        # Accumulated by calls to self.push.
        # These are of the form [[trajectory_0: time_0, time_1,...], [trajectory_1: time_0, time_1, ...], ...].
        # These have shape (number of trajectories, trajectory length).
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

        # Computed in self.end_trajectory.
        # This has the same shape as above, (number of trajectories, trajectory length).
        self.returns = []
        self.advantages = [] # Normalized in self._finish_buffer.
        # These have shape (number of trajectories,).
        self.episode_reward_sums = []
        self.episode_lengths = []

        # This is the number of stored experiences, i.e. the number of calls to self.push.
        self.size = 0

        # Prepare for calls to self.push.
        self._start_trajectory()


    def _start_trajectory(self):
        """
        Add new empty trajectory sequences for states, actions, values, & rewards.
        """
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        self.values.append([])


    def __len__(self):
        """
        Return the number of experiences = the number of calls to self.push.
        """
        return self.size


    def push(self, state, action, reward, value):
        """
        Store information about this policy step.
        Set self.ready_to_sample to False.

        Parameters
        ----------
        state : np.ndarray
            The observed state.
        action : np.ndarray
            The action taken in that state.
        reward : float
            The instantaneous reward for this state and action.
        value : float
            The estimated value of that state.
        # done : bool
        #     Whether this action led to a terminal state.
        """
        # self._finish_buffer must be called after the last call to self.push before calling self.sample.
        assert not self.ready_to_sample, "Call to self.push after self._finish_buffer."

        self.states[-1].append(state)
        self.actions[-1].append(action)
        self.rewards[-1].append(reward)
        self.values[-1].append(value)

        self.size += 1


    def end_trajectory(self, terminal_value):
        """
        Finish the most recently stored trajectory by computing its returns, advantages, episode_reward_sum, and episode_length.

        This should be called at the end of each episode.

        Parameters
        ----------
        terminal_value : float
            The estimated future discounted returns of the last state reached, 0 if that state was terminal.
        """
        trajectory_rewards = self.rewards[-1]
        assert len(trajectory_rewards) > 0

        # Iterate backwards through trajectory_rewards and assign the return at time step t as reward_t + discount_factor_gamma * return_t_+_1.
        R = terminal_value
        returns = []
        for reward in reversed(trajectory_rewards):
            R = reward + self.discount_factor_gamma * R
            # Appending and reversing should be faster than returns.insert(0, R).
            returns.append(R)
        returns.reverse()
        self.returns.append(returns)

        # Compute advantages.
        if self.GAE_lambda == 1:
            # Use shortcut equivalent to GAE with lambda = 1.
            values = np.array(self.values[-1])
            returns = np.array(self.returns[-1])
            advantages = list(returns - values)
        else:
            # Use GAE (Generalized Advantage Estimate).
            # Based on: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
            rewards = self.rewards[-1]
            values = self.values[-1]
            values_with_terminal_value = values + [terminal_value]
            advantages = [0] # For recursive base case, trimmed at the end.
            for t in reversed(range(len(self.returns))):
                delta = rewards[t] + self.discount_factor_gamma * values_with_terminal_value[t] - values_with_terminal_value[t + 1]
                advantages.append(delta + self.GAE_lambda * self.discount_factor_gamma * advantages[-1])
            del advantages[0]
            advantages.reverse()
        self.advantages.append(advantages)

        # Compute episode diagnostics.
        self.episode_reward_sums.append(np.sum(trajectory_rewards))
        self.episode_lengths.append(len(trajectory_rewards))

        # This prepares self.states, self.actions, self.rewards, & self.values for pushes as part of a new trajectory.
        self._start_trajectory()

    
    def _finish_buffer(self):
        """
        Prepare for a call to self.sample by normalizing advantages and trimming empty sequences from self._start_trajectory.
        Cannot call self.push after self._finish_buffer until self.reset.

        Must be called after ending the last trajectory.
        """
        # Assert buffer is nonempty.
        assert len(self.states) > 0, "No states have been pushed yet."
        # Assert the latest trajectory has been ended.
        assert len(self.returns) == len(self.states), "self.end_trajectory has not been called on the latest trajectory."
        # Assert that self._start_trajectory has been called since the last push.
        assert len(self.states[-1]) == len(self.actions[-1]) == len(self.rewards[-1]) == len(self.values[-1]) == 0, \
            "self._start_trajectory has not been called since the last push."
        
        # Trim empty sequences from self._start_trajectory.
        del self.states[-1]
        del self.actions[-1]
        del self.rewards[-1]
        del self.values[-1]

        # Normalize advantages.
        flattened_advantages = np.array(list(itertools.chain.from_iterable(self.advantages)))
        # flattened_advantages = np.concatenate(self.advantages) # Equivalent, yet slower.
        advantages_mean = np.mean(flattened_advantages)
        advantages_std = np.std(flattened_advantages)
        # TODO: find a more graceful way to avoid division by zero. Maybe an adaptive addition in the denominator.
        self.advantages = [list((np.array(advantages_trajectory) - advantages_mean) / (advantages_std + 1e-4)) for advantages_trajectory in self.advantages]
        
        self.ready_to_sample = True


    def sample(self, get_whole_trajectories, mini_batch_size):
        """
        Return a generator that yields randomly sampled mini-batches as (states, actions, returns, advantages, not_padding_mask) tuples.

        If get_whole_trajectories == False:
            Each element is a torch.Tensor with shape (mini_batch_size,).
            states has shape (mini_batch_size, <state_size>).

        elif get_whole_trajectories == True:
            Each element is a torch.Tensor with shape (mini_batch_size, <longest_trajectory_length>).
            states has shape (mini_batch_size, <longest_trajectory_length>, <state_size>).
            The <trajectory_length> dimension is padded to the length of the longest trajectory in the mini-batch.


        Parameters
        ----------
        get_whole_trajectories : bool
            If True, sample whole trajectories of experiences. If False, sample individual experiences.
            This should be True for recurrent policies.
        mini_batch_size : int
            The number of experience tuples per mini-batch.
        """
        if not self.ready_to_sample:
            # self._finish_buffer sets self.ready_to_sample to True.
            self._finish_buffer()
        assert self.ready_to_sample
        
        if not get_whole_trajectories:
            # Sample individual experiences.
        
            # Flatten experiences and cast to torch.Tensor.
            flattened_states = torch.Tensor(np.concatenate(self.states)) # Retains the <state_size> dimension at the end.
            flattened_actions = torch.Tensor(np.concatenate(self.actions))
            flattened_returns = torch.Tensor(np.concatenate(self.returns))
            flattened_advantages = torch.Tensor(np.concatenate(self.advantages))
            # These went from shape (<number_of_trajectories>, <trajectory_length>, <state_size for states>)
            # to shape (<number_of_experiences>, <state_size for states>).

            # Get mini-batches of random indices of experience tuples.
            randomized_experience_indices = SubsetRandomSampler(range(len(flattened_states)))
            mini_batch_indices_generator = BatchSampler(randomized_experience_indices, mini_batch_size, drop_last=True)

            # Yield mini-batches of experience tuples.
            for mini_batch_indices in mini_batch_indices_generator:
                # Select the experiences for this mini-batch.
                states_batch = flattened_states[mini_batch_indices]
                actions_batch = flattened_actions[mini_batch_indices]
                returns_batch = flattened_returns[mini_batch_indices]
                advantages_batch = flattened_advantages[mini_batch_indices]

                not_padding_mask = 1 # There is no padding when get_whole_trajectories == False.

                yield states_batch, actions_batch, returns_batch, advantages_batch, not_padding_mask
        
        else: # get_whole_trajectories == True
            # Sample whole trajectories of experiences.

            # Get mini-batches of random indices of trajectories.
            randomized_trajectory_indices = SubsetRandomSampler(range(len(flattened_states)))
            mini_batch_indices_generator = BatchSampler(randomized_trajectory_indices, mini_batch_size, drop_last=True)

            # Yield mini-batches of whole trajectories.
            for mini_batch_indices in mini_batch_indices_generator:
                # Select the trajectories for this mini-batch.
                states_batch     = list(np.array(self.states    , dtype=object)[mini_batch_indices])
                actions_batch    = list(np.array(self.actions   , dtype=object)[mini_batch_indices])
                returns_batch    = list(np.array(self.returns   , dtype=object)[mini_batch_indices])
                advantages_batch = list(np.array(self.advantages, dtype=object)[mini_batch_indices])
                
                # Get not_padding_mask: ones where the actual ragged trajectories come from real experiences.
                not_padding_mask = [np.ones_like(returns_trajectory) for returns_trajectory in returns_batch]

                # Pad experience trajectories and cast to torch.Tensor.
                states_batch     = pad_sequence([torch.Tensor(state_trajectory    ) for state_trajectory     in self.states_batch    ], batch_first=False, padding_value=0.0)
                actions_batch    = pad_sequence([torch.Tensor(action_trajectory   ) for action_trajectory    in self.actions_batch   ], batch_first=False, padding_value=0.0)
                returns_batch    = pad_sequence([torch.Tensor(return_trajectory   ) for return_trajectory    in self.returns_batch   ], batch_first=False, padding_value=0.0)
                advantages_batch = pad_sequence([torch.Tensor(advantage_trajectory) for advantage_trajectory in self.advantages_batch], batch_first=False, padding_value=0.0)
                # Pad not_padding_mask to match the padded experiences.
                not_padding_mask = pad_sequence(not_padding_mask, batch_first=False, padding_value=0.0)
                # These went from shape (<mini_batch_size>, <trajectory_length>, <state_size for states>)
                # to shape (<mini_batch_size>, <longest_trajectory_length>, <state_size for states>).

                yield states_batch, actions_batch, returns_batch, advantages_batch, not_padding_mask


    @classmethod
    def merge_buffers(cls, buffers, discount_factor_gamma=None, GAE_lambda=None):
        
        # Assert buffers have not tried to sample yet.
        for buffer in buffers:
            assert not buffer.ready_to_sample, "buffer.ready_to_sample should be False for all buffers, \
                else they may have already attempted to sample and have called _finish_buffer. \
                This is a problem because we want to call _finish_buffer to normalize advantages after merge_buffers."
        
        # Get buffer attributes.
        discount_factor_gamma = buffers[0].discount_factor_gamma
        GAE_lambda = buffers[0].GAE_lambda
        for buffer in buffers:
            assert buffer.discount_factor_gamma == discount_factor_gamma and buffer.GAE_lambda == GAE_lambda, "All buffers must have the same discount_factor_gamma and GAE_lambda attributes."
        
        # Create new buffer object.
        merged_buffer = cls(discount_factor_gamma, GAE_lambda)
        
        # Trim empty sequences from self._start_trajectory.
        del merged_buffer.states[-1]
        del merged_buffer.actions[-1]
        del merged_buffer.rewards[-1]
        del merged_buffer.values[-1]

        for buffer in buffers:
            # Copy pushed experiences.
            merged_buffer.states.extend(buffer.states)
            merged_buffer.states.extend(buffer.states)
            merged_buffer.states.extend(buffer.states)
            merged_buffer.states.extend(buffer.states)
            # Copy derived values.
            merged_buffer.returns.extend(buffer.returns)
            merged_buffer.advantages.extend(buffer.advantages)
            # Copy diagnostics.
            merged_buffer.episode_reward_sums.extend(buffer.episode_reward_sums)
            merged_buffer.episode_lengths.extend(buffer.episode_lengths)
            # Accumulate buffer sizes.
            merged_buffer.size += buffer.size
            
        return merged_buffer