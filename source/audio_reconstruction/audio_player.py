import librosa
import numpy as np
import pandas as pd
import torch
import sounddevice as sd
from scipy.interpolate import CubicSpline

from source.config import AudioPlayerConfig


class AudioPlayer:

    def __init__(self):
        self.config = AudioPlayerConfig()
        vocoder, _, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                                                                'nvidia_hifigan')
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

        mel_spectrogram = self.convert_to_mel(log_spectrogram, self.config.sr, self.config.n_fft, self.config.n_mels)
        if self.config.sr < self.config.gan_sr:
            mel_spectrogram = self.upsample_spectrogram(mel_spectrogram, self.config.sr, self.config.gan_sr)

        mel_spectrogram = self.apply_log_compression(mel_spectrogram, self.config.compression_factor)
        mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            audio = self.vocoder(mel_spectrogram).float()
            audio = self.denoiser(audio.squeeze(1), self.config.denoising_strength)
            audio = audio.squeeze().cpu().numpy()

        audio = librosa.util.normalize(audio)

        if self.config.mode == 'play':
            sd.play(audio, self.config.gan_sr)
            sd.wait()
        elif self.config.mode == 'return':
            return audio, self.config.gan_sr

    def apply_log_compression(self, mel_spectrogram, C):
        return np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None) * C)

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
