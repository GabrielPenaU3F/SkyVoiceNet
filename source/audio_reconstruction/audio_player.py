import librosa
import numpy as np
import pandas as pd
import torch
import sounddevice as sd

from source.config import AudioPlayerConfig
from source.data_processing.spectrogram_transformer import SpectrogramTransformer


class AudioPlayer:

    def __init__(self):
        self.config = AudioPlayerConfig()
        self.transformer = SpectrogramTransformer()

    def play_audio_from_spectrogram(self, log_spectrogram, **kwargs):
        self.config.update(**kwargs)

        if isinstance(log_spectrogram, pd.Series):
            log_spectrogram = log_spectrogram.to_numpy()
        elif isinstance(log_spectrogram, torch.Tensor):
            log_spectrogram = log_spectrogram.cpu().detach().numpy()

        audio = self.griffin_lim(log_spectrogram)
        if self.config.mode == 'play':
            sd.play(audio, self.config.sr)
            sd.wait()
        elif self.config.mode == 'return':
            return audio, self.config.sr

    def griffin_lim(self, log_spectrogram):
        log_spectrogram = np.pad(log_spectrogram, ((0, 1), (0, 0)), mode='constant') # Pad the frequency we removed
        magnitude_spectrogram = self.transformer.obtain_magnitude_spectrogram(log_spectrogram)
        magnitude_spectrogram = np.power(magnitude_spectrogram, 1.2)
        return librosa.griffinlim(magnitude_spectrogram, n_iter=32, hop_length=256, win_length=1024, window='hann')
