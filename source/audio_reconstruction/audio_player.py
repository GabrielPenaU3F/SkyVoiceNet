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
        vocoder, _, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                                                'nvidia_hifigan')
        self.transformer = SpectrogramTransformer()
        self.vocoder = vocoder
        self.vocoder.eval()
        self.vocoder = self.vocoder.to(self.config.device)
        self.denoiser = denoiser.to(self.config.device)

    def play_audio_from_spectrogram(self, log_spectrogram, **kwargs):
        self.config.update(**kwargs)

        if isinstance(log_spectrogram, pd.Series):
            log_spectrogram = log_spectrogram.to_numpy()
        elif isinstance(log_spectrogram, torch.Tensor):
            log_spectrogram = log_spectrogram.cpu().detach().numpy()

        if self.config.method == 'hifi-gan':

            audio = self.hifi_gan(log_spectrogram)
            if self.config.mode == 'play':
                sd.play(audio, self.config.gan_sr)
                sd.wait()
            elif self.config.mode == 'return':
                return audio, self.config.gan_sr

        elif self.config.method == 'griffin-lim':

            audio = self.griffin_lim(log_spectrogram)
            if self.config.mode == 'play':
                sd.play(audio, self.config.sr)
                sd.wait()
            elif self.config.mode == 'return':
                return audio, self.config.sr


    def hifi_gan(self, log_spectrogram):
        mel_spectrogram = self.transformer.convert_to_mel(log_spectrogram, self.config.sr, self.config.n_fft,
                                                          self.config.n_mels)
        if self.config.sr < self.config.gan_sr:
            mel_spectrogram = self.transformer.upsample_spectrogram(mel_spectrogram, self.config.sr, self.config.gan_sr)
        mel_spectrogram = self.apply_log_compression(mel_spectrogram, self.config.compression_factor)
        mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            audio = self.vocoder(mel_spectrogram).float()
            audio = self.denoiser(audio.squeeze(1), self.config.denoising_strength)
            audio = audio.squeeze().cpu().numpy()
        return audio

    def apply_log_compression(self, mel_spectrogram, C):
        return np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None) * C)

    def griffin_lim(self, log_spectrogram):
        log_spectrogram = np.pad(log_spectrogram, ((0, 1), (0, 0)), mode='constant') # Pad the frequency we removed
        magnitude_spectrogram = self.transformer.obtain_magnitude_spectrogram(log_spectrogram)
        return librosa.griffinlim(magnitude_spectrogram, n_iter=32, hop_length=256, win_length=1024, window='hann')
