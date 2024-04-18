import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    """
    The base class which all model/policy networks inherit from. It includes methods
    for normalizing states.
    """
    def __init__(self):
        super(Net, self).__init__()

        # Params for nn-input normalization
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1


    def normalize_state(self, state: torch.Tensor, update_normalization=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1)).to(state.device)
            self.welford_state_mean_diff = torch.ones(state.size(-1)).to(state.device)

        if update_normalization:
            assert len(state.size()) == 1 # we expect to get a single state vector
            state_old = self.welford_state_mean
            self.welford_state_mean += (state - state_old) / self.welford_state_n
            self.welford_state_mean_diff += (state - state_old) * (state - state_old)
            self.welford_state_n += 1

        normalized_state = (state - self.welford_state_mean) / torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)
        return normalized_state


    def copy_normalizer_stats_from_net(self, net):
        self.welford_state_mean      = net.welford_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n         = net.welford_state_n

    def initialize_parameters(self):
        raise NotImplementedError
        # self.apply(normc_fn)
        if hasattr(self, 'critic_last_layer'):
            self.critic_last_layer.weight.data.mul_(0.01)


    def _base_forward(self, x):
        raise NotImplementedError


def create_layers(layer_fn, input_dim, layer_sizes):
    """
    This function creates a pytorch modulelist and appends
    pytorch modules like nn.Linear or nn.LSTMCell passed
    in through the layer_fn argument, using the sizes
    specified in the layer_sizes list.
    """
    ret = nn.ModuleList()
    ret += [layer_fn(input_dim, layer_sizes[0])]
    for i in range(len(layer_sizes)-1):
        ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
    return ret


def get_activation(act_name):
    try:
        return getattr(torch, act_name)
    except:
        raise RuntimeError(f"Not implemented activation {act_name}. Please add in.")


class FFBase(Net):
    """
    The base class for feedforward networks.
    """
    def __init__(self, in_dim, layer_sizes, nonlinearity='tanh'):
        super(FFBase, self).__init__()
        self.layers       = create_layers(nn.Linear, in_dim, layer_sizes)
        self.nonlinearity = get_activation(nonlinearity)

    def _base_forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = self.nonlinearity(layer(x))
        return x


class LSTMBase(Net):
    """
    The base class for LSTM networks.
    """
    def __init__(self, in_dim, layer_sizes):
        super(LSTMBase, self).__init__()
        self.layer_sizes = layer_sizes
        for layer_size in self.layer_sizes:
            assert layer_size == self.layer_sizes[0], "LSTMBase only supports layer_sizes of equal size"
            # I think because LSTM has this constraint, LSTMCell may be used for more flexibility
        self.lstm = nn.LSTM(in_dim, self.layer_sizes[0], len(self.layer_sizes))
        self.init_hidden_state()

    def init_hidden_state(self, **kwargs):
        self.h_c = None

    def get_hidden_state(self):
        return self.h_c[0], self.h_c[1]

    def set_hidden_state(self, hidden, cells):
        self.h_c = (hidden, cells)

    def _base_forward(self, x):
        dims = x.dim()
        if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
            x = x.view(1, -1)
        elif dims == 3:
            self.init_hidden_state()

        x, self.h_c = self.lstm(x, self.h_c)

        if dims == 1:
            x = x.view(-1)

        return x