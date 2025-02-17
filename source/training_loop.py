import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, loss_fn, dataset, batch_size, num_epochs, learning_rate,
                lr_patience=5, lr_reduction=0.5, plateau_threshold=1e-2, device='cuda'):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduction, patience=lr_patience, threshold=plateau_threshold)

    model.to(device)
    training_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        torch.autograd.set_detect_anomaly(True)
        model.train()

        for batch_idx, (speech_spec, contour_spec, target_spec) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

            speech_spec = speech_spec.to(device)
            contour_spec = contour_spec.to(device)
            target_spec = target_spec.to(device)

            optimizer.zero_grad()

            # Forward pass
            predicted_spec = model(speech_spec, contour_spec)

            # Backprop
            loss = loss_fn(predicted_spec, target_spec)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            epoch_loss += loss.item()

            # For memory usage optimization
            del loss, predicted_spec, target_spec # Delete tensors
            torch.cuda.empty_cache() # Clear memory
            torch.cuda.synchronize()

        avg_loss = epoch_loss / len(dataloader)
        training_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model, training_loss
