import librosa
import numpy as np
from scipy.ndimage import zoom


class SpectrogramTransformer:

    def obtain_log_spectrogram(self, audio, n_fft, hop_length, win_length):
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length = hop_length, win_length=win_length)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
        return spectrogram_db

    def stretch_spectrogram(self, spectrogram, target_length):
        scale_factor = target_length / spectrogram.shape[1]
        stretched_spectrogram = zoom(spectrogram, (1, scale_factor), order=1)
        return stretched_spectrogram

    def zeropad_time(self, spectrogram, target_length, padding='zero'):
        _, t = spectrogram.shape
        pad_t = target_length - t
        if padding == 'min':
            min_value = np.min(spectrogram)
            padding = min_value
        elif padding == 'zero':
            padding = 0
        else:
            return spectrogram
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_t)), mode='constant', constant_values=padding)
        return spectrogram
