import os

import numpy as np
import sounddevice as sd
import librosa
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.network.min_max_wrapper import MinMaxWrapper

from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import draw_spectrograms, draw_single_spectrogram, compute_spectrogram_energy

path_dir = PathRepo().get_output_path()
model_file = os.path.join(path_dir, 'sky_voice_net_4_full_cat_60_batch4.pt')

trained_model = torch.load(model_file)

# Load test data
dataset = DataLoader.load_processed_data('reduced_dataset_4.h5', dataset=None)
sample = dataset.sample(n=1)
speech_spectrogram = sample.iloc[0]['speech']
melody_spectrogram = sample.iloc[0]['song']
melody_contour = sample.iloc[0]['contour']

# Shape [batch, channel, freq, time]
speech_spectrogram_tensor = torch.tensor(speech_spectrogram).unsqueeze(0).unsqueeze(0).to('cuda').float()
melody_contour_tensor = torch.tensor(melody_contour).unsqueeze(0).unsqueeze(0).to('cuda').float()

# Predict
net = MinMaxWrapper(trained_model)
net.eval()
with torch.no_grad():
    predicted_spectrogram = net(speech_spectrogram_tensor, melody_contour_tensor).squeeze()

# Draw
predicted_spectrogram = predicted_spectrogram.cpu().numpy()
draw_spectrograms(speech_spectrogram, melody_spectrogram, 'Speech spectrogram', 'Melody spectrogram')
draw_single_spectrogram(predicted_spectrogram, 'Predicted')

print("Range (real):", np.min(melody_spectrogram), np.max(melody_spectrogram))
print("Range (predicted):", np.min(predicted_spectrogram), np.max(predicted_spectrogram))

real_total_energy, real_energy_per_frame = compute_spectrogram_energy(melody_spectrogram)
pred_total_energy, pred_energy_per_frame = compute_spectrogram_energy(predicted_spectrogram)

print(f"Total energy (real): {real_total_energy}, Total energy (predicted): {pred_total_energy}")

plt.plot(real_energy_per_frame, label="Real", alpha=0.7)
plt.plot(pred_energy_per_frame, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Framewise spectrogram energy")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.show()

# Reconstruct
player = AudioPlayer()
# player.play_audio_from_spectrogram(predicted_spectrogram, sr=16000, compression_factor=3, denoising_strength=0.2)

# Save
original_audio, new_sr = player.play_audio_from_spectrogram(melody_spectrogram, sr=16000, method='griffin-lim',
                                                     compression_factor=3, denoising_strength=0.2, mode='return')
predicted_audio, _ = player.play_audio_from_spectrogram(predicted_spectrogram, sr=16000, method='griffin-lim',
                                                     compression_factor=3, denoising_strength=0.2, mode='return')
path = PathRepo().get_test_wads_path()
original_audio = (original_audio * 32767).astype(np.int16)
predicted_audio = (predicted_audio * 32767).astype(np.int16)
wavfile.write(os.path.join(path, 'ex4_original_full_60_batch4.wav'), new_sr, original_audio)
wavfile.write(os.path.join(path, 'ex4_predicted_full_60_batch4.wav'), new_sr, predicted_audio)
