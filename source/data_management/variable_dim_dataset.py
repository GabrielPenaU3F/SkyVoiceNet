import numpy as np
import torch
from torch.utils.data import Dataset


class VariableLengthDataset(Dataset):

    def __init__(self, data):
        """
        data:
            - 'speech'
            - 'contour'
            - 'song'
        """
        self.data = data  # List of dicts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        speech_spec = torch.tensor(sample['speech'], dtype=torch.float32).unsqueeze(0)
        contour_spec = torch.tensor(sample['contour'], dtype=torch.float32).unsqueeze(0)
        target_spec = torch.tensor(sample['song'], dtype=torch.float32).unsqueeze(0)

        return speech_spec, contour_spec, target_spec

    def train_test_split(self, test_size, seed=None):

        if seed is not None:
            np.random.seed(seed)

        n = len(self.data)
        test_indices = np.random.choice(len(self.data), size=test_size, replace=False)
        train_indices = list(set(range(n)) - set(test_indices))

        train_data = [self.data[i] for i in train_indices]
        test_data = [self.data[i] for i in test_indices]
        train_set = VariableLengthDataset(train_data)
        test_set = VariableLengthDataset(test_data)

        return train_set, test_set
