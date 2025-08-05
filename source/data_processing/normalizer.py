import librosa
import numpy as np
import torch
import torch.nn.functional as F


class Normalizer:

    @staticmethod
    def normalize(audio):
        return librosa.util.normalize(audio)

    @staticmethod
    def minmax_normalize_batch(batch):
        # batch: tensor de shape (B, F, T)
        B = batch.size(0)
        normed = torch.zeros_like(batch)
        mins = torch.zeros(B, 1, 1, device=batch.device)
        maxs = torch.zeros(B, 1, 1, device=batch.device)

        for i in range(B):
            x = batch[i]
            x_min = x.min()
            x_max = x.max()
            normed[i] = (x - x_min) / (x_max - x_min + 1e-8)
            mins[i] = x_min
            maxs[i] = x_max

        return normed, mins, maxs

    @staticmethod
    def minmax_denormalize(spectrogram, mins, maxs):
        # batch_normed: shape (B, F, T), mins and maxs: shape (B, 1, 1)
        T = spectrogram.size(3)
        padded_mins = torch.stack([Normalizer.pad_tensor(tensor, T) for tensor in mins])
        padded_maxs = torch.stack([Normalizer.pad_tensor(tensor, T) for tensor in maxs])
        return spectrogram * (padded_maxs - padded_mins + 1e-8) + padded_mins

    @staticmethod
    def pad_tensor(tensor, length):
        return F.pad(tensor, (0, length - tensor.size(2)), "constant", 0)