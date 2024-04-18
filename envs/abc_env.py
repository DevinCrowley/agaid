from abc import ABC, ABCMeta, abstractmethod

from ..sim import get_sim, ABC_Sim


class ABC_Env:
    """Abstract Base Class Env: this is the base class that instantiable envs (environments) should inherit from."""
    

    @abstractmethod
    def  __init__(self, **kwargs):
        """Assign self.sim to an ABC_Sim subclass."""
        # TODO: redo this sim attribute code.
        if 'sim' in kwargs:
            sim_class = get_sim(kwargs['sim'])
        else:
            default_sim_class = ABC_Sim # In subclasses, assign this to a default sim class for this env.
            sim_class = get_sim(default_sim_class)
        # This assumes all sim-specific kwargs are in kwargs.
        self.sim = sim_class(**kwargs)
        # self.sim = get_module_attribute('..sims', config['sim'])
    
    
    @abstractmethod
    def reset(self):
        """Reset episode. Call self.sim.reset(). Return observation_state from sim_state."""
        sim_state = self.sim.reset()
        observation_state = sim_state
        return observation_state
    
    
    def step(self, action):
        """Call self.simulation_forward(action). Return observation_state, reward, done, info."""
        sim_state = self.simulation_forward(action, self.num_sim_steps)
        # Compute observation_state.
        observation_state = self.get_observation_state(sim_state)
        reward = self.compute_reward()
        done = self.compute_done()
        info = dict()
        return observation_state, reward, done, info

    
    @abstractmethod
    def get_observation_state(self, sim_state):
        """
        Compute the full observed state, used as input to the policy.

        Parameters
        ----------
        sim_state : _type_
            The state information from self.sim.

        Returns
        -------
        _type_
            The full observed state, used as input to the policy.
        """
        return sim_state
    

    @abstractmethod
    def compute_reward(self):
        pass
    

    @abstractmethod
    def compute_done(self):
        pass
    
    
    # def seed(self, seed=None):
    #     rng, seed = gym.utils.seeding.np_random(seed)
    #     raise NotImplementedError
    
    
    # def render(self, mode='human'):
    #     raise NotImplementedError


    def simulation_forward(self, action, num_sim_steps):
        for i in range(num_sim_steps):
            sim_state = self.sim.step(action)
        return sim_state


    @property
    @abstractmethod
    def observation_size(self):
        return self._observation_size