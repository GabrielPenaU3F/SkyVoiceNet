import os

import matplotlib.pyplot as plt
import torch

from source.data_management.path_repo import PathRepo
from source.network.sky_voice_net import SkyVoiceNet

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SkyVoiceNet().to(device)
outputs_path = PathRepo().get_output_path()
checkpoint = torch.load(os.path.join(outputs_path, "sky_voice_net.pth"), map_location=device)

for name, module in model.named_modules():
    print(name)

model.eval()


attn_weights = model.attention_block.attention_layer.attn_weights
attn_weights = attn_weights.detach().cpu().numpy()

plt.imshow(attn_weights, aspect='auto', cmap='viridis')
plt.colorbar(label="Attention Weight")
plt.xlabel("Key Positions")
plt.ylabel("Query Positions")
plt.title("Attention Weights Heatmap")
plt.show()