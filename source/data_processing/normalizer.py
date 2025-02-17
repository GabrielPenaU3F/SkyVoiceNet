import librosa
import numpy as np


class Normalizer:

    def normalize(self, audio):
        return librosa.util.normalize(audio)

    def spectrogram_min_max_normalize(self, spectrogram):
        min_val = np.min(spectrogram)
        max_val = np.max(spectrogram)
        return (spectrogram - min_val) / (max_val - min_val)
