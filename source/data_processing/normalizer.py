import torch
import torch.nn.functional as F


class Normalizer:

    @staticmethod
    def minmax_normalize_batch(batch):
        # batch: tensor with shape (B, F, T)
        B = batch.size(0)
        normalized = torch.zeros_like(batch)
        mins = torch.zeros(B, 1, 1, device=batch.device)
        maxs = torch.zeros(B, 1, 1, device=batch.device)

        for i in range(B):
            x = batch[i]
            x_min = x.min()
            x_max = x.max()
            normalized[i] = (x - x_min) / (x_max - x_min + 1e-8)
            mins[i] = x_min
            maxs[i] = x_max

        return normalized

    @staticmethod
    def pad_tensor(tensor, length):
        return F.pad(tensor, (0, length - tensor.size(2)), "constant", 0)