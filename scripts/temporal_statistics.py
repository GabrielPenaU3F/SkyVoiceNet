import os

import numpy as np
from matplotlib import pyplot as plt

from source.data_management.path_repo import PathRepo

path = os.path.join(PathRepo().get_base_path(), 'outputs', 'statistics')
data = np.load(os.path.join(path, 'temporal_dimensions.npy'))

print(f'Mean = {np.mean(data):.4f}')
print(f'STD = {np.std(data):.4f}')
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins='fd', density=True, color='b', alpha=0.75)
ax.set_xlabel('Duración (en frames)', fontsize=18)
ax.set_ylabel('Probabilidad', fontsize=18)
ax.tick_params(axis='both', labelsize=14)
fig.tight_layout()

fig.savefig('time_frames_histogram.pdf')
plt.show()