import librosa
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import zoom


class SpectrogramTransformer:

    def obtain_log_spectrogram(self, audio, n_fft, hop_length, win_length):
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        spectrogram = np.delete(spectrogram, -1, axis=0) # Delete highest frequency so we have just 512
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

    def obtain_magnitude_spectrogram(self, log_spectrogram):
        return librosa.db_to_amplitude(log_spectrogram)

    def convert_to_mel(self, y, sr, n_fft, n_mels):
        magnitude_spectrogram = np.exp(y) - 1
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, fmin=0, fmax=8000, n_mels=n_mels)
        mel_basis = np.delete(mel_basis, -1, 1) # Delete highest frequency
        return np.dot(mel_basis, magnitude_spectrogram)

    def upsample_spectrogram(self, spectrogram, orig_sr=16000, target_sr=22050):
        factor = target_sr / orig_sr
        time_steps = np.linspace(0, spectrogram.shape[1] - 1, num=int(spectrogram.shape[1] * factor))

        upsampled = np.zeros((spectrogram.shape[0], len(time_steps)))
        for i in range(spectrogram.shape[0]):
            spline = CubicSpline(np.arange(spectrogram.shape[1]), spectrogram[i])
            upsampled[i] = spline(time_steps)

        return upsampled.astype(np.float32)
