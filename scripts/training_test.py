import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.data_processing.normalizer import Normalizer

from source.network.sky_voice_net import SkyVoiceNet
from source.training_loop import train_model


dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=True)

net = SkyVoiceNet(conv_out_channels=64)

# def activation_hook(module, input, output):
#     print(f"{module.__class__.__name__}: mean={output.mean().item()}, std={output.std().item()}, min={output.min().item()}, max={output.max().item()}")
#
# for name, module in net.named_modules():
#     if isinstance(module, (nn.Conv2d, nn.Linear, nn.TransformerEncoder, nn.TransformerDecoder, nn.BatchNorm2d,
#                            nn.LayerNorm, nn.ConvTranspose2d)):
#         module.register_forward_hook(activation_hook)

# Parameters
batch_size = 4
num_epochs = 4
learning_rate = 2e-2
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
loss_fn = torch.nn.MSELoss()

# Train
trained_model, training_loss = train_model(
    model=net,
    loss_fn=loss_fn,
    dataset=dataset,
    batch_size=batch_size,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device,
)
#
path_dir = PathRepo().get_output_path()
output_file = os.path.join(path_dir, 'sky_voice_net.pt')
torch.save(trained_model, output_file)

fig, ax = plt.subplots()
t = np.arange(1, num_epochs + 1)
ax.plot(t, training_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Loss evolution over time")
output_png = os.path.join(path_dir, 'loss.png')

fig.savefig(output_png, dpi=300)
plt.show()
