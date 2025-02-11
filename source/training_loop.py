import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataset, batch_size, num_epochs, learning_rate, device='cuda'):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

            # Paso forward: obtener la predicción de la red
            predicted_spec = model(speech_spec, contour_spec)  # Aquí pasamos los inputs de forma correcta

            # Calcular pérdida
            loss = loss_f(predicted_spec, target_spec)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model, training_loss
