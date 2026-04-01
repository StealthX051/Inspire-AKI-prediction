import torch
from torch.utils.data import Dataset
import vitaldb
import numpy as np

class VitalDBDataset(Dataset):
    def __init__(self, file_path, track_names, interval=None, transform=None):
        """
        Args:
            file_path (str): Path to the VitalDB file.
            track_names (list): List of track names to load.
            interval (float): Interval for sampling. If None, uses maximum resolution.
            transform (callable, optional): Optional transform to apply on data.
        """
        self.file_path = file_path
        self.track_names = track_names
        self.interval = interval
        self.transform = transform

        # Load data from VitalDB file
        self.vf = vitaldb.VitalFile(file_path, track_names)
        
        # Convert to numpy array for efficient loading in __getitem__
        self.data = self.vf.to_numpy(track_names, interval=interval)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Get the sample and apply transformations if needed
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # Convert sample to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        
        return sample