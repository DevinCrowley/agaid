from pathlib import Path
import pickle

import numpy as np


class Offline_Data_IO:
    """
    This class is for writing and reading experiences for offline RL.
    """

    def __init__(self, data_dir, create_nonexistent_data_dir=True):
        self.data_dir = Path(data_dir).expanduser().resolve()

        if create_nonexistent_data_dir and not self.data_dir.is_dir():
            self.data_dir.mkdir()
        
        if not self.data_dir.is_dir():
            raise RuntimeError(f"self.data_dir {self.data_dir} is not an existing directory.")

    
    def write(self, episode_data):
        # Validate inputs.
        # Verify episode_data is a dict.
        assert isinstance(episode_data, dict)
        # Verify episode_data has exactly the expected keys.
        assert episode_data.keys() == {'obs', 'action', 'next_obs', 'done'}
        # Validate each value as a 1D numpy array of appropriate dtype.
        for key, val in episode_data.items():
            episode_data[key] = np.array(val).astype(bool if key == 'done' else float)
            assert episode_data[key].ndim == 1
        # Verify each value has the same length.
        key_1_array = np.array(episode_data.keys())
        key_2_array = np.roll(key_1_array, -1)
        [assert len(episode_data[key_1]) == len(episode_data[key_2]) for key_1, key_2 in zip(key_1_array, key_2_array)]
        # episode_data is a dict mapping keys to equal length 1D numpy arrays.

        # Generate file_name.
        # TODO: dynamically generate unique file_name with hyperparameter hash.
        base_name = "episode"
        index = 0
        suffix = ".pkl"
        file_name = f"{base_name}_({index}){suffix}"
        file_path = self.data_dir / file_name
        while file_path.is_file():
            index += 1
            file_name = f"{base_name}_({index}){suffix}"
            file_path = self.data_dir / file_name

        # Save data.
        with open(file_path, 'wb') as file:
            pickle.dump(file, episode_data)
        

    
    def read(self, file_name):
        # Verify inputs.
        file_path = self.data_dir / file_name
        if not file_path.is_file():
            raise RuntimeError(f"file_path {file_path} is not an existing file.")

        # Read data.
        with open(file_path, 'rb') as file:
            return pickle.load(file)