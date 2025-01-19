import librosa
import numpy as np
import torch
import torchcrepe
from matplotlib import pyplot as plt

from source import utilities

hop_length = 256

melody, fs = torchcrepe.load.audio('resources/NUS-48E/nus-smc-corpus_48/ADIZ/sing/01.wav')
melody_np = melody.numpy().squeeze()
melody_trim, melody_idx = librosa.effects.trim(melody_np, top_db=40, hop_length=hop_length)
melody_tensor = torch.tensor(melody_trim)
melody_tensor /= torch.max(torch.abs(melody_tensor))
melody_cuda = melody_tensor.to(device='cuda')[None, :]

contour = torchcrepe.predict(melody_cuda, fs, hop_length=hop_length, model='full', device='cuda:0',
                             fmin=50.0, fmax=1500.0, batch_size=2048)
win_length = 5  # Ventana de 5 frames (~25 ms)
smooth_contour = torchcrepe.filter.median(contour, win_length)
midi_contour = utilities.convert_to_midi_note(smooth_contour)

# utilities.play_midi_notes(midi_contour.cpu().numpy().squeeze(), hop_length/fs, fs)

# Calcula los tiempos del contorno
midi_contour = midi_contour.cpu().numpy().squeeze()
num_frames = len(midi_contour)
time_indices = np.arange(num_frames) * hop_length / fs

# Gráfico del contorno MIDI
plt.subplot(2, 1, 1)
plt.plot(time_indices, midi_contour, marker='.', linestyle='None', color='r', label='Contorno MIDI')
plt.title("Contorno MIDI")
plt.xlabel("Tiempo (s)")
plt.ylabel("Número MIDI")
plt.legend()

# Espectrograma de la señal de audio original
D = librosa.amplitude_to_db(np.abs(librosa.stft(melody.numpy().squeeze(), hop_length=hop_length)), ref=np.max)
plt.subplot(2, 1,2)
librosa.display.specshow(D, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title("Espectrograma de la señal de audio original")
plt.xlabel("Tiempo (s)")
plt.ylabel("Frecuencia (Hz)")

plt.tight_layout()
plt.show()
