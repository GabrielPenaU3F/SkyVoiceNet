import torch
from torch.utils.data import Dataset


class VariableLengthDataset(Dataset)\
        :
    def __init__(self, data):
        """
        data: lista de diccionarios con las claves:
            - 'speech_spectrogram'
            - 'contour_spectrogram'
            - 'target_spectrogram'
        """
        self.data = data  # Lista de diccionarios

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        speech_spec = torch.tensor(sample['speech_spectrogram'], dtype=torch.float32).unsqueeze(0)
        contour_spec = torch.tensor(sample['contour'], dtype=torch.float32).unsqueeze(0)
        target_spec = torch.tensor(sample['melody_spectrogram'], dtype=torch.float32).unsqueeze(0)

        return speech_spec, contour_spec, target_spec