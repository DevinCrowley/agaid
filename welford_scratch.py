
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