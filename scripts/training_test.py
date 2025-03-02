import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

from source.network.sky_voice_net import SkyVoiceNet
from source.training_loop import train_model


dataset = DataLoader.load_processed_data('nus_processed_zeroed.h5', as_tensor_dataset=True)

net = SkyVoiceNet(conv_out_channels=64)

# Parameters
batch_size = 5
num_epochs = 5
learning_rate = 1e-4
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
