import os

import numpy as np
import pandas as pd
from scipy.io import wavfile

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

path = PathRepo().get_base_path()
test_audios = DataLoader.load_test_data('test_pred_audios.h5')
filename = os.path.join(PathRepo().get_hdf5_path(), 'metrics_all.h5')
df = pd.read_hdf(filename, key="metrics")
df5 = df[df["model"] == "model_5"]
pesqs_5 = df5["PESQ"].to_numpy()

# Índices
idx_min = np.argmin(pesqs_5)
idx_max = np.argmax(pesqs_5)
idx_med = np.argmin(np.abs(pesqs_5 - pesqs_5.mean()))

# Extract audios

audio_min_orig = test_audios.data[idx_min]['original']
audio_min_pred5 = test_audios.data[idx_min]['pred_5']

audio_max_orig = test_audios.data[idx_max]['original']
audio_max_pred5 = test_audios.data[idx_max]['pred_5']

audio_med_orig = test_audios.data[idx_med]['original']
audio_med_pred5 = test_audios.data[idx_med]['pred_5']

# Format

audio_min_orig = (audio_min_orig * 32767).astype(np.int16)
audio_min_pred5 = (audio_min_pred5 * 32767).astype(np.int16)

audio_max_orig = (audio_max_orig * 32767).astype(np.int16)
audio_max_pred5 = (audio_max_pred5 * 32767).astype(np.int16)

audio_med_orig = (audio_med_orig * 32767).astype(np.int16)
audio_med_pred5 = (audio_med_pred5 * 32767).astype(np.int16)

# Write

wavfile.write(os.path.join(path, 'outputs', 'statistics', 'min_pesq_orig.wav'), 16000, audio_min_orig)
wavfile.write(os.path.join(path, 'outputs', 'statistics', 'min_pesq_pred5.wav'), 16000, audio_min_pred5)

wavfile.write(os.path.join(path, 'outputs', 'statistics', 'max_pesq_orig.wav'), 16000, audio_max_orig)
wavfile.write(os.path.join(path, 'outputs', 'statistics', 'max_pesq_pred5.wav'), 16000, audio_max_pred5)

wavfile.write(os.path.join(path, 'outputs', 'statistics', 'med_pesq_orig.wav'), 16000, audio_med_orig)
wavfile.write(os.path.join(path, 'outputs', 'statistics', 'med_pesq_pred5.wav'), 16000, audio_med_pred5)
