"""
Copied and partially modified from r3/ppo.py
"""
if 1:
    raise NotImplementedError

from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from time import monotonic

import ipdb as pdb #  for debugging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
import ray

from .mt_model_buffer import MT_Model_Buffer
from .dynamics_models import Dynamics_Model_Multihead, Dynamics_Model_Embed, Dynamics_Model_Aggregate

"""
TODO:
Make abstract buffer class
Pass normalization params from buffer to policy
Wrap actor & critic in a single Policy class
Give actor & critic networks a reset method that calls init_hidden_state if available
Expand done to trunkated and terminated
Separate a privileged_state for critic input
"""

class Actor_Critic_Mixin:
    """
        Mixin class for maintaining copies of an actor and critic nn.
        Supports syncing parameters.

        Args:
            actor: actor pytorch network
            critic: critic pytorch network
            device: device to store worker's networks on, cuda or cpu. None defaults to cpu.
    """
    def __init__(self, actor, critic, device=None):
        self.actor = deepcopy(actor).to(device)
        self.critic = deepcopy(critic).to(device)
        assert not (self.actor.recurrent ^ self.critic.recurrent), f"actor & critic expected to share recurrent status."\
                                                                   f"actor.recurrent: {actor.recurrent}, critic.recurrent: {critic.recurrent}"
        if self.actor.recurrent:
            self.recurrent = True
        else:
            self.recurrent = False

    def sync_policy(self, new_actor_params, new_critic_params, input_norm=None):
        """
        Function to sync the actor and critic parameters with new parameters.

        Args:
            new_actor_params (torch dictionary): New actor parameters to copy over
            new_critic_params (torch dictionary): New critic parameters to copy over
            input_norm (int): Running counter of states for normalization
        """
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)

        if input_norm is not None:
            raise NotImplementedError
            self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n = input_norm
            self.critic.copy_normalizer_stats(self.actor)


@ray.remote
class Sampler(Actor_Critic_Mixin):
    """
    Worker for sampling experience online.

    Args:
        actor: actor pytorch network
        critic: critic pytorch network
        env_factory: environment constructor function
        gamma: discount factor

    Attributes:
        env: instance of environment
        gamma: discount factor
        dynamics_randomization: if dynamics_randomization is enabled in environment
    """
    def __init__(self, actor, critic, env, gamma, worker_id, device=None):
        super().__init__(self, actor=actor, critic=critic, device=device)
        self.gamma     = gamma
        self.env       = env
        self.worker_id = worker_id


    def sample_episode(self,
                    max_steps: int = 300,
                    do_eval: bool = False,
                    update_normalization_param: bool=False):
        """
        Function to sample experience.

        Args:
            max_steps: maximum trajectory length of an episode
            do_eval: if True, use deterministic policy
            update_normalization_param: if True, update normalization parameters
        """
        start_t = monotonic()
        torch.set_num_threads(1)
        # Toggle models to eval mode
        self.actor.eval()
        self.critic.eval()
        memory = Buffer(self.gamma)
        with torch.no_grad():
            state = torch.Tensor(self.env.reset())
            done = False
            value = 0
            episode_step = 0
            info_dict = {}

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state()
            if hasattr(self.critic, 'init_hidden_state'):
                self.critic.init_hidden_state()

            while not done and episode_step < max_steps:
                state = torch.Tensor(state)
                if hasattr(self.env, 'get_privilege_state') and self.critic.use_privilege_critic:
                    privilege_state = self.env.get_privilege_state()
                    critic_state = privilege_state
                    info_dict['privilege_states'] = privilege_state
                else:
                    critic_state = state
                action = self.actor(state,
                                    deterministic=do_eval,
                                    update_normalization_param=update_normalization_param)

                # Get value based on the current critic state (s)
                if do_eval:
                    # If is evaluation, don't need critic value
                    value = 0.0
                else:
                    value = self.critic(torch.Tensor(critic_state)).numpy()

                next_state, reward, done, _ = self.env.step(action.numpy())

                reward = np.array([reward])
                memory.push(state.numpy(), action.numpy(), reward, value)
                # If environment has additional info, push additional info to buffer
                if hasattr(self.env, 'get_info_dict'):
                    memory.push_additional_info(self.env.get_info_dict())

                state = next_state
                if hasattr(self.env, 'get_privilege_state') and self.critic.use_privilege_critic:
                    critic_state = self.env.get_privilege_state()
                else:
                    critic_state = state
                episode_step += 1

            # Compute the terminal value based on the state (s')
            value = (not done) * self.critic(torch.Tensor(critic_state)).numpy()
            memory.end_trajectory(terminal_value=value)

        return memory, episode_step / (monotonic() - start_t), self.worker_id




class PPO(Actor_Critic_Mixin):
    """
    Worker for sampling experience for PPO

    Args:
        actor: actor pytorch network
        critic: critic pytorch network
        env_factory: environment constructor function
        args: argparse namespace

    Attributes:
        actor: actor pytorch network
        critic: critic pytorch network
        recurrent: recurrent policies or not
        env_factory: environment constructor function
        gamma: discount factor
        entropy_coeff: entropy regularization coeff

        grad_clip: value to clip gradients at
        mirror: scalar multiple of mirror loss. No mirror loss if this equals 0
        env: instance of environment
        state_mirror_indices (func): environment-specific function for mirroring state for mirror loss
        action_mirror_indices (func): environment-specific function for mirroring action for mirror loss
        workers (list): list of ray worker IDs for sampling experience
        optim: ray woker ID for optimizing
    """

    def __init__(self,
        actor,
        critic,
        env_factory,
        *,
        gamma,
        gae_lambda,
        entropy_coeff,
        grad_clip,
        eval_freq,
        workers,
        redis=None,
        device=None,
    ):
        super().__init__(self, actor=actor, critic=critic, device=device) # Defines self.actor & self.critic.

        self.env_factory   = env_factory
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.grad_clip     = grad_clip
        self.env_factory   = env_factory
        self.eval_freq     = eval_freq

        if not ray.is_initialized():
            if redis is not None:
                ray.init(redis_address=redis)
            else:
                ray.init(num_cpus=workers)

        self.workers = [AlgoSampler.remote(actor, critic, env_factory, args.gamma, i) for i in \
                        range(args.workers)]
        self.optimizer = PPOOptim(actor, critic, mirror_dict, **vars(args))