from abc import ABC, abstractmethod


class ABC_Buffer(ABC):
    """Abstract Base Class Buffer: this is the base class that instantiable buffers should inherit from."""
    

    @abstractmethod
    def __len__(self):
        pass


    @abstractmethod
    def __init__(self):
        self.ready_to_sample = False
        self.reset()


    @abstractmethod
    def reset(self):
        self.ready_to_sample = False
        pass


    @abstractmethod
    def push(self):
        self.ready_to_sample = False
        pass


    @abstractmethod
    def end_trajectory(self):
        pass

    
    @abstractmethod
    def _finish_buffer(self):
        self.ready_to_sample = True
        pass


    @abstractmethod
    def sample(self, batch_size):
        if not self.ready_to_sample:
            self._prepare_buffer()
        pass


    @classmethod
    @abstractmethod
    def merge_buffers(cls, buffers):
        pass