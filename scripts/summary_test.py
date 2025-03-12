import os

import torch

from source.data_management.path_repo import PathRepo
from source.utilities import count_parameters

path_dir = PathRepo().get_output_path()
output_file = os.path.join(path_dir, 'sky_voice_net.pt')
model = torch.load(output_file)

count_parameters(model)
