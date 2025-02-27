import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, loss_fn, dataset, batch_size, num_epochs, learning_rate, gamma=0.9, device='cuda'):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    # scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduction, patience=lr_patience, threshold=plateau_threshold)

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

            # # Visualize gradients
            # gradients = []
            #
            # for param in model.parameters():
            #     if param.grad is not None:
            #         gradients.append(param.grad.view(-1).cpu().detach().numpy())
            #
            # plt.hist(np.concatenate(gradients), bins=100)
            # plt.yscale("log")
            # plt.xlabel("Gradiente")
            # plt.ylabel("Frecuencia")
            # plt.title("Distribución de Gradientes")
            # plt.show()

            optimizer.step()
            scheduler.step()
            # scheduler.step(loss)

            epoch_loss += loss.item()

            # For memory usage optimization
            del loss, predicted_spec, target_spec # Delete tensors
            torch.cuda.empty_cache() # Clear memory
            torch.cuda.synchronize()

        avg_loss = epoch_loss / len(dataloader)
        training_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model, training_loss
