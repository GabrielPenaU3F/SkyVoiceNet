import librosa
import numpy as np
import torch
import torchcrepe
from matplotlib import pyplot as plt

from source import utilities

import numpy as np
import torch
import torch.nn.functional as F

from source.data_processing.normalizer import Normalizer


def build_contour_spectrogram(contour_hz, spec_shape, sr=44100):
    """
    Construye una imagen binaria de tamaño (num_bins x T) con el contorno de la melodía de CREPE.

    Parámetros:
      - contour_hz: (T_c,) Array con las frecuencias de la melodía en Hz (salida de CREPE).
      - spec_shape: (num_bins, T_s) Dimensiones del espectrograma al que queremos ajustarnos.
      - sr: Frecuencia de muestreo del audio (default: 16000 Hz).
      - num_bins: Número de bins de frecuencia del espectrograma.

    Retorna:
      - contour_spectrogram: (num_bins, T_s) Imagen binaria con la melodía marcada.
    """
    num_bins_spec, num_frames_spec = spec_shape  # Dimensiones del espectrograma

    # Convertimos el contorno a escala MIDI
    midi_contour = 69 + 12 * torch.log2(pitch_contour / 440)
    midi_contour = torch.clamp(torch.round(midi_contour), 0, 127).to(torch.float32)
    midi_contour = F.interpolate(midi_contour.unsqueeze(0), num_frames_spec, mode='nearest')
    midi_contour = midi_contour.cpu().numpy().squeeze().squeeze()

    # 3. Creamos la imagen binaria
    contour_spectrogram = np.zeros((num_bins_spec, num_frames_spec), dtype=np.float32)

    # Llenamos la imagen binaria
    for t_idx, midi_note in enumerate(midi_contour):
        bin_idx = midi_to_stft_bin(midi_note, sr, 1024)
        contour_spectrogram[bin_idx, t_idx] = 1  # Marcamos la frecuencia correspondiente

    return contour_spectrogram


# Mapeo MIDI -> bins de frecuencia en la STFT
def midi_to_stft_bin(midi_note, sr, n_fft):
    """Convierte un número MIDI a un índice de bin de la STFT."""
    num_bins = n_fft // 2 + 1
    freq_hz = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))  # Convertimos MIDI a Hz
    bin_idx = int(round((freq_hz / (sr / 2)) * (n_fft // 2)))  # Convertimos Hz a índice STFT
    return np.clip(bin_idx, 0, num_bins - 1)  # Aseguramos que el índice esté en el rango


melody, fs = torchcrepe.load.audio('../resources/NUS-48E/nus-smc-corpus_48/ADIZ/sing/01.wav')
melody_np = melody.numpy().squeeze()
melody_trim, melody_idx = librosa.effects.trim(melody_np, top_db=40, hop_length=256)
melody_trim = melody_trim / np.max(np.abs(melody_trim))
melody_tensor = torch.from_numpy(melody_trim)[None, :].to(device='cuda')

pitch_contour = torchcrepe.predict(melody_tensor, fs, hop_length=256, model='full', device='cuda:0',
                             fmin=50.0, fmax=1500.0, batch_size=2048)

spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(
    melody_tensor.cpu().numpy().squeeze(), hop_length=256, n_fft=1024, win_length=1024)), ref=np.max)
spectrogram = np.delete(spectrogram, -1, axis=0)  # Delete highest frequency so we have just 512
spectrogram = Normalizer.normalize(spectrogram)

# win_length = 5  # Ventana de 5 frames (~25 ms)
# smooth_contour = torchcrepe.filter.median(pitch_contour, 1024)
contour = build_contour_spectrogram(pitch_contour, spectrogram.shape, fs)
print(np.min(contour), np.max(contour))

# utilities.play_midi_notes(midi_contour.cpu().numpy().squeeze(), hop_length/fs, fs)

# Gráfico del contorno MIDI
utilities.draw_spectrograms(contour, spectrogram, 'Contour', 'Melody')
