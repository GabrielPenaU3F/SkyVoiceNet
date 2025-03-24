import os
import random

import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile
from torch.utils.data import Subset

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.evaluation import evaluate_model
from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import draw_spectrograms, draw_single_spectrogram, compute_spectrogram_energy

net = SkyVoiceNet()

path_dir = PathRepo().get_output_path()
output_file = os.path.join(path_dir, 'sky_voice_net_full_cat_30.pt')

trained_model = torch.load(output_file)
dataset = DataLoader.load_processed_data('nus_processed_2.h5', dataset='variable')

torch.manual_seed(42)
training_set, test_set = dataset.train_test_split(578, seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()
batch_size = 4

test_loss = evaluate_model(net, dataset,
                           loss_fn=loss_fn,
                           device=device,
                           batch_size=batch_size)

print(f"Total loss in test set: {test_loss:.6f}")

# Pick two random examples

random.seed(42)
selected_indices = random.sample(range(len(dataset)), 2)
samples = Subset(dataset, selected_indices)

speech_1, contour_1, song_1 = samples[0]
speech_2, contour_2, song_2 = samples[1]

with torch.no_grad():
    predicted_1 = trained_model(speech_1.unsqueeze(0).to(device), contour_1.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    predicted_2 = trained_model(speech_2.unsqueeze(0).to(device), contour_2.unsqueeze(0).to(device)).squeeze().cpu().numpy()


# # Example 1

speech_1 = speech_1.squeeze().numpy()
song_1 = song_1.squeeze().numpy()
speech_2 = speech_2.squeeze().numpy()
song_2 = song_2.squeeze().numpy()

draw_spectrograms(speech_1, song_1, 'Speech spectrogram (EX7)', 'Melody spectrogram (EX7)')
draw_single_spectrogram(predicted_1, 'Predicted spectrogram (EX7)')

real_total_energy_1, real_energy_per_frame_1 = compute_spectrogram_energy(song_1)
pred_total_energy_1, pred_energy_per_frame_1 = compute_spectrogram_energy(predicted_1)

print(f"Total energy (real): {real_total_energy_1}, Total energy (predicted): {pred_total_energy_1}")

plt.plot(real_energy_per_frame_1, label="Real", alpha=0.7)
plt.plot(pred_energy_per_frame_1, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Framewise spectrogram energy (EX7)")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.show()

# # Example 2

draw_spectrograms(speech_2, song_2, 'Speech spectrogram (EX8)', 'Melody spectrogram (EX8)')
draw_single_spectrogram(predicted_2, 'Predicted spectrogram (EX8)')

real_total_energy_2, real_energy_per_frame_2 = compute_spectrogram_energy(song_2)
pred_total_energy_2, pred_energy_per_frame_2 = compute_spectrogram_energy(predicted_2)

print(f"Total energy (real): {real_total_energy_2}, Total energy (predicted): {pred_total_energy_2}")

plt.plot(real_energy_per_frame_2, label="Real", alpha=0.7)
plt.plot(pred_energy_per_frame_2, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Framewise spectrogram energy (EX8)")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.show()

# Reconstruct
player = AudioPlayer()
audio_1, sr_1 = player.play_audio_from_spectrogram(predicted_1, sr=16000, compression_factor=3, denoising_strength=0.05, mode='return')
audio_2, sr_2 = player.play_audio_from_spectrogram(predicted_2, sr=16000, compression_factor=3, denoising_strength=0.05, mode='return')

wavfile.write('../outputs/full/Example 7.wav', sr_1, audio_1)
wavfile.write('../outputs/full/Example 8.wav', sr_2, audio_2)
