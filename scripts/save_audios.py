import os

import numpy as np
from scipy.io import wavfile

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

preprocessed_data = DataLoader.load_preprocessed_data('nus_preprocessed_2.h5')

sr = 16000

for i in range(100):
    melody = (np.array(preprocessed_data['song'][i]) * 32767).astype(np.int16)
    speech = (np.array(preprocessed_data['speech'][i]) * 32767).astype(np.int16)
    melody_filename = 'melody_' + str(i) + '.wav'
    speech_filename = 'speech_' + str(i) + '.wav'
    path = PathRepo().get_test_wads_path()
    wavfile.write(os.path.join(path, melody_filename), sr, melody)
    wavfile.write(os.path.join(path, speech_filename), sr, speech)
