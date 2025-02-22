import librosa
import numpy as np
import torch
import sounddevice as sd
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader

def preprocess_mel(mel, C):
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None) * C)
    return mel

raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)

audio = raw_data.iloc[0]['sing']

# vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
vocoder, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
vocoder.eval()
vocoder = vocoder.to('cuda')
denoiser = denoiser.to('cuda')

audio_sr = 44100
sr = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
C = 1
denoising_strength = 0.005

audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=sr)
stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=80, fmin=0, fmax=8000)
mel_spectrogram = np.dot(mel_basis, stft)

mel_spectrogram = preprocess_mel(mel_spectrogram, C=C)

plt.imshow(mel_spectrogram, aspect="auto", origin='lower', cmap='magma')
plt.colorbar(format='%+2.0f', label="Mel")
plt.title("Singing voice (Mel)")
plt.xlabel("Time (s)")
plt.ylabel("Freq")
plt.show()

# Inference
mel_spectrogram_tensor = torch.tensor(mel_spectrogram).unsqueeze(0).to('cuda')
with torch.no_grad():
    # audio = vocoder(mel_spectrogram_tensor).squeeze().cpu().numpy()
    audio = vocoder(mel_spectrogram_tensor).float()
    audio = denoiser(audio.squeeze(1), denoising_strength)
    audio = audio.squeeze().cpu().numpy()

sd.play(audio, sr)
sd.wait()
