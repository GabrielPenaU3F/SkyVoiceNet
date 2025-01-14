import librosa
import sounddevice as sd
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader
from source.data_management.data_preprocessor import DataPreprocessor

data_loader = DataLoader()
proc = DataPreprocessor()

raw_data = data_loader.load_data('nus_data_raw.h5', load_wavs=False)
processed_data = proc.preprocess(raw_data, save=False)

# raw_speech_ex = raw_data['read'].iloc[0]
# silenceless_speech_ex = processed_data['speech'].iloc[0]
#
# sd.play(silenceless_speech_ex, 44100)
# sd.wait()

# processed_data = data_loader.load_data('nus_processed.h5')
#
# row = processed_data.iloc[0]
# sing_spec = row['sing']
#
# fig, ax = plt.subplots()
# img = librosa.display.specshow(sing_spec, ax=ax, cmap='autumn_r')
# fig.colorbar(img, ax=ax)
#
# plt.show()