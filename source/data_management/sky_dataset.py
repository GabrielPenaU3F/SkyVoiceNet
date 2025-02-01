import numpy as np
import torch
from torch.utils.data import Dataset


class SkyVoiceDataset(Dataset):

    def __init__(self, speech_series, contour_series, melody_series, speech_sample_rates, melody_sample_rate):

        assert len(speech_series) == len(contour_series) == len(melody_series) == len(speech_sample_rates)

        self.speech_spec_tensor = self.series_to_tensor(speech_series) # Tensor
        self.contour_spec_tensor = self.series_to_tensor(contour_series) # Tensor
        self.melody_spec_tensor = self.series_to_tensor(melody_series) # Tensor
        self.speech_sample_rates = speech_sample_rates # List
        self.melody_sample_rate = melody_sample_rate # List (possibly constant)

    def series_to_tensor(self, series):
        return torch.tensor(np.stack(series)[:, None, :, :], dtype=torch.float32)

    def __len__(self):
        return len(self.speech_spec_tensor)

    def __getitem__(self, idx):
        return (self.speech_spec_tensor[idx],
                self.contour_spec_tensor[idx],
                self.melody_spec_tensor[idx],
                self.speech_sample_rates[idx],
                self.melody_sample_rate[idx])
