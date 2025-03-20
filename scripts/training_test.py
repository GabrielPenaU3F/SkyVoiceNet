import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

from source.network.sky_voice_net import SkyVoiceNet
from source.training_loop import train_model
from source.utilities import count_parameters

dataset = DataLoader.load_processed_data('nus_processed_2.h5', dataset='variable')
# dataset = DataLoader.load_processed_data('reduced_dataset_2.h5', dataset='variable')

net = SkyVoiceNet(mode='cat_pre_post_attn')

torch.manual_seed(42)

# Train-test split
training_set, test_set = dataset.train_test_split(578, seed=42)

# Full dataset in training
# training_set = dataset

# Parameters
batch_size = 16
num_epochs = 90
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
loss_fn = torch.nn.MSELoss()

# Train
trained_model, training_loss = train_model(
    model=net,
    loss_fn=loss_fn,
    dataset=training_set,
    batch_size=batch_size,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    device=device,
)

count_parameters(trained_model)

path_dir = PathRepo().get_output_path()
output_file = os.path.join(path_dir, 'sky_voice_net_full_pre_post_attn.pt')
torch.save(trained_model, output_file)

fig, ax = plt.subplots()
t = np.arange(1, num_epochs + 1)
ax.plot(t, training_loss)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.set_title("Loss evolution over time")
output_png = os.path.join(path_dir, 'loss_full_cat_pre_post_attn.png')

fig.savefig(output_png, dpi=300)
plt.show()
