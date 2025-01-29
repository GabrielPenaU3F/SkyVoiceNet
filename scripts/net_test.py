import numpy as np
import torch
from matplotlib import pyplot as plt

from source import utilities
from source.data_management.data_loader import DataLoader
from source.data_processing.processing_pipeline import ProcessingPipeline
from torch.utils.data import TensorDataset

from source.network.convolutional_block import ConvolutionalBlock
from source.training_loop import train_model


processed_data = DataLoader.load_processed_data('nus_processed.h5')

speech_spec = processed_data['speech_spectrogram']
melody_spec = processed_data['melody_spectrogram']

speech_tensors = torch.tensor(
    np.stack([np.expand_dims(arr, axis=0) for arr in speech_spec.values]),
    dtype=torch.float32
)
target_tensors = torch.tensor(
    np.stack([np.expand_dims(arr, axis=0) for arr in melody_spec.values]),
    dtype=torch.float32
)

dataset = TensorDataset(speech_spec, melody_spec)
net = ConvolutionalBlock(in_channels=1, out_channels=32)

# Parámetros
batch_size = 16
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