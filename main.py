import librosa
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader
from source.data_management.data_preprocessor import DataPreprocessor

data_loader = DataLoader()
proc = DataPreprocessor()

raw_data = data_loader.load_data('nus_data_raw.h5', load_wavs=True)
processed_data = proc.preprocess(raw_data, save=True)

processed_data = data_loader.load_data('nus_processed.h5')

row = processed_data.iloc[0]
sing_spec = row['sing']

fig, ax = plt.subplots()
img = librosa.display.specshow(sing_spec, ax=ax, cmap='autumn_r')
fig.colorbar(img, ax=ax)

plt.show()