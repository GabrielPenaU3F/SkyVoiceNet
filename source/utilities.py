import librosa
import numpy as np
import torch
from matplotlib import pyplot as plt


def series_to_tensor(series):
    return torch.tensor(np.stack(series)[:, None, :, :], dtype=torch.float32)

def draw_spectrograms(spectrogram_1, spectrogram_2, title_1='', title_2='', sr=16000):

    num_bins, num_frames = spectrogram_1.shape  # Extraemos dimensiones
    freqs = np.linspace(0, sr/2, num_bins)  # Frecuencias en Hz
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    im_1 = ax[0].imshow(spectrogram_1, aspect="auto", origin='lower', cmap='magma', extent=[0, num_frames, freqs[0], freqs[-1]])
    ax[0].set_title(title_1)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Freq")
    fig.colorbar(im_1, format='%+2.0f ', label="Intensity")

    num_bins, num_frames = spectrogram_2.shape  # Extraemos dimensiones
    freqs = np.linspace(0, sr/2, num_bins)  # Frecuencias en Hz
    im_2 = ax[1].imshow(spectrogram_2, aspect="auto", origin='lower', cmap='magma', extent=[0, num_frames, freqs[0], freqs[-1]])
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Freq")
    ax[1].set_title(title_2)
    fig.colorbar(im_2, label="Intensity")

    fig.tight_layout()
    plt.show()

def draw_single_spectrogram(spectrogram, title='', sr=16000):
    num_bins, num_frames = spectrogram.shape  # Extraemos dimensiones
    freqs = np.linspace(0, sr/2, num_bins)  # Frecuencias en Hz

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(spectrogram, aspect="auto", origin='lower', cmap='magma', extent=[0, num_frames, freqs[0], freqs[-1]])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq")
    fig.colorbar(im, label="Intensity")

    fig.tight_layout()
    plt.show()

def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total Trainable Parameters:", total_params)

def compute_spectrogram_energy(log_spectrogram):
    spectrogram = librosa.db_to_amplitude(log_spectrogram)
    total_energy = np.sum(spectrogram**2)
    energy_per_frame = np.sum(spectrogram**2, axis=0)

    return total_energy, energy_per_frame

def match_spectrogram_dims(spec_1, spec_2, spec_3):
    min_len = min(spec_1.shape[-1], spec_2.shape[-1], spec_3.shape[-1])
    spec_1 = spec_1[..., :min_len]
    spec_2 = spec_2[..., :min_len]
    spec_3 = spec_3[..., :min_len]
    return spec_1, spec_2, spec_3
