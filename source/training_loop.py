import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.data_processing.normalizer import Normalizer
# from source.network.sky_voice_net import SkyVoiceNet


def train_model(model, loss_fn, dataset, batch_size, num_epochs, learning_rate, device='cuda'):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.9), eps=1e-9)

    model = model.to(device)
    training_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        torch.autograd.set_detect_anomaly(True)
        model.train()

        for batch_idx, (speech_spec, melody_contour, melody_spec) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

            speech_spec = speech_spec.to(device)
            melody_contour = melody_contour.to(device)
            target_spec = melody_spec.to(device)

            # Normalize target
            target_spec = Normalizer.minmax_normalize_batch(target_spec)

            optimizer.zero_grad()

            # Forward pass
            predicted_spec = model(speech_spec, melody_contour)

            # Backprop
            loss = loss_fn(predicted_spec, target_spec)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} gradient norm: {param.grad.norm().item()}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_loss += loss.item()

            # For memory usage optimization
            del loss, predicted_spec, target_spec # Delete tensors
            torch.cuda.empty_cache() # Clear memory
            torch.cuda.synchronize()

        avg_loss = epoch_loss / len(dataloader)
        training_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model, training_loss


def collate_fn(batch):
    spectrograms, contours, targets = zip(*batch)

    if len(spectrograms) > 1:
        # Maximum length is rounded up to the closest multiple of 8
        max_length = max(tensor.size(2) for tensor in spectrograms)
        max_length = (max_length + 7) // 8 * 8

        # Zero-pad all tensors
        spectrograms = [pad_tensor(tensor, max_length) for tensor in spectrograms]
        contours = [pad_tensor(tensor, max_length) for tensor in contours]
        targets = [pad_tensor(tensor, max_length) for tensor in targets]

    return torch.stack(spectrograms), torch.stack(contours), torch.stack(targets)

def pad_tensor(tensor, length):
    return F.pad(tensor, (0, length - tensor.size(2)), "constant", 0)