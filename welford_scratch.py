import numpy as np
import torch


class Welford_Norm_Mixin:
    
    def  __init__(self):
        self._welford_state_count = 0
        self._welford_state_mean = torch.zeros(1) # Shape of state.
        self._welford_state_var_sum = torch.ones(1) # Shape of state.

    def update_welford_state_norm(self, state):
        """Update welford normalization stats."""

        assert state.dim() == 1

        # Broadcast welford mean and var_sum for state.
        assert self._welford_state_mean.size() == self._welford_state_var_sum.size()
        if self._welford_state_mean.size() != state.size():
            assert torch.equal(self._welford_state_mean, torch.zeros(1)) and torch.equal(self._welford_state_var_sum, torch.ones(1))
            assert self._welford_state_count == 0
            self._welford_state_mean = torch.zeros_like(state)
            self._welford_state_var_sum = torch.ones_like(state)
        
        # Update welford stats for state.
        self._welford_state_count += 1
        delta = state - self._welford_state_mean
        self._welford_state_mean += delta / self._welford_state_count
        delta2 = state - self._welford_state_mean
        self._welford_state_var_sum += delta * delta2

    def update_welford_state_norm_batch(self, state_batch):
        """Update welford normalization stats. Assumes last dimension is states."""
        
        if state_batch.dim() > 2:
            state_batch = state_batch.view(-1, state_batch.size()[-1])
        assert state_batch.dim() ==  2
        state_size = state_batch.size()[-1]

        state_batch_mean = torch.mean(state_batch, dim=0)
        batch_size = len(state_batch)
        # batch_m2 = sum((x - batch_mean) * (x - batch_mean) for x in state_batch)

        # Broadcast welford mean and var_sum for state.
        assert self._welford_state_mean.size() == self._welford_state_var_sum.size()
        if self._welford_state_mean.size() != state_size:
            assert torch.equal(self._welford_state_mean, torch.zeros(1)) and torch.equal(self._welford_state_var_sum, torch.ones(1))
            assert self._welford_state_count == 0
            self._welford_state_mean = torch.zeros(state_size)
            self._welford_state_var_sum = torch.ones(state_size)
        
        # Update welford stats for state.
        #  TODO: fix
        self._welford_state_count += batch_size
        delta = state_batch_mean - self._welford_state_mean
        self._welford_state_mean += delta * batch_size / self._welford_state_count
        delta2 = state_batch_mean - self._welford_state_mean
        self._welford_state_var_sum += delta * delta2 * batch_size * (self._welford_state_count-batch_size)/self._welford_state_count




    @staticmethod
    def _normalize(value, welford_count, welford_mean, welford_var_sum, inplace=False):
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
            value[...] = torch.div(torch.sub(value, welford_mean), stdev)
        else:
            value[...] = (value - welford_mean) / stdev
        return value
    def normalize_state(self, state, inplace=False):
        return self._normalize(value=state, welford_count=self._welford_state_count, welford_mean=self._welford_state_mean, welford_var_sum=self._welford_state_var_sum, inplace=inplace)
    # def normalize_action(self, action, inplace=False):
    #     return self._normalize(value=action, welford_count=self._welford_action_count, welford_mean=self._welford_action_mean, welford_var_sum=self._welford_action_var_sum, inplace=inplace)

    
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
        return self._denormalize(value=state, welford_count=self._welford_state_count, welford_mean=self._welford_state_mean, welford_var_sum=self._welford_state_var_sum, inplace=inplace)
    # def denormalize_action(self, action, inplace=False):
    #     return self._denormalize(value=action, welford_count=self._welford_action_count, welford_mean=self._welford_action_mean, welford_var_sum=self._welford_action_var_sum, inplace=inplace)





    def normalize_state(self, state: torch.Tensor, update_normalization_param=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1)).to(state.device)
            self.welford_state_mean_diff = torch.ones(state.size(-1)).to(state.device)

        if update_normalization_param:
            if len(state.size()) == 1:  # if we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - state_old)
                self.welford_state_n += 1
            else:
                raise RuntimeError  # this really should not happen
        return (state - self.welford_state_mean) / torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean      = net.welford_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n         = net.welford_state_n





    def reset():
        
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.welford_state_count = 0
        self.welford_state_mean = None # torch.zeros(1) # Shape of state.
        self.welford_state_var_sum = None # torch.ones(1) # Shape of state.
        self.welford_action_count = 0
        self.welford_action_mean = None # torch.zeros(1) # Shape of action.
        self.welford_action_var_sum = None # torch.ones(1) # Shape of action.


    def  push():

        # Update welford normalization stats.
        
        self.welford_state_count += 1
        self.welford_action_count += 1

        # Broadcast welford means and var_sums for state.
        state = self.states[-1][-1]
        if self.welford_state_mean is None or self.welford_state_var_sum is None:
            assert self.welford_state_mean is self.welford_state_var_sum is None
            assert self.size == self.welford_state_count == 1
            self.welford_state_mean = torch.zeros_like(state)
            self.welford_state_var_sum = torch.ones_like(state)
        
        # Broadcast welford means and var_sums for action.
        action = self.actions[-1][-1]
        if self.welford_action_mean is None or self.welford_action_var_sum is None:
            assert self.welford_action_mean is self.welford_action_var_sum is None
            assert self.size == self.welford_action_count == 1
            self.welford_action_mean = torch.zeros_like(action)
            self.welford_action_var_sum = torch.ones_like(action)
        
        # Update welford stats for state.
        delta = state - self.welford_state_mean
        self.welford_state_mean += delta / self.welford_state_count
        delta2 = state - self.welford_state_mean
        self.welford_state_var_sum += delta * delta2
        
        # Update welford stats for action.
        delta = action - self.welford_action_mean
        self.welford_action_mean += delta / self.welford_action_count
        delta2 = action - self.welford_action_mean
        self.welford_action_var_sum += delta * delta2

    
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