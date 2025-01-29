import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataset, batch_size, num_epochs, learning_rate, device):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_f = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    training_loss = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()

        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_f(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        training_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    return model, training_loss
