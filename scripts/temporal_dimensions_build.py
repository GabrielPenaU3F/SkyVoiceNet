import os

import numpy as np

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

dataset = DataLoader.load_processed_data('nus_processed_7.h5', dataset=None)

temporal_dims = []
speechs = dataset['speech']

for spectrogram in speechs.items():
    spec = np.array(spectrogram[1])
    temporal_dims.append(spec.shape[-1])

path = os.path.join(PathRepo().get_base_path(), 'outputs', 'statistics')
np.save(os.path.join(path, 'temporal_dimensions.npy'), np.array(temporal_dims))
