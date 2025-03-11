import librosa
import numpy as np


class Normalizer:

    @staticmethod
    def normalize(audio):
        return librosa.util.normalize(audio)

    @staticmethod
    def spectrogram_min_max_normalize(spectrogram):
        min_val = np.min(spectrogram)
        max_val = np.max(spectrogram)
        return (spectrogram - min_val) / (max_val - min_val)
