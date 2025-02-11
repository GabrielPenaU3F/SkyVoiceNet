
import torch
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader

from source.network.sky_voice_net import SkyVoiceNet
from source.training_loop import train_model


dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=True)

net = SkyVoiceNet(conv_out_channels=64)

# Parámetros
batch_size = 8
num_epochs = 10
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"  # Usar GPU si está disponible

# Entrenar el modelo
trained_model, training_loss = train_model(
    model=net,
    dataset=dataset,
    batch_size=batch_size,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device
)

plt.plot(training_loss)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Evolución de la pérdida durante el entrenamiento")
plt.show()