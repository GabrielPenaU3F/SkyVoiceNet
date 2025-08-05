import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import wavfile
from torch.utils.data import Subset

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.evaluation import evaluate_model
from source.network.min_max_wrapper import MinMaxWrapper
from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import draw_spectrograms, draw_single_spectrogram, compute_spectrogram_energy

path_dir = PathRepo().get_output_path()
model_file = os.path.join(path_dir, 'sky_voice_net_4_full_cat_60_batch4.pt')

trained_model = torch.load(model_file)
dataset = DataLoader.load_processed_data('nus_processed_4.h5', dataset='variable')

torch.manual_seed(42)
training_set, test_set = dataset.train_test_split(578, seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()
batch_size = 4

test_loss = evaluate_model(trained_model, dataset,
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

net = MinMaxWrapper(trained_model)
net.eval()
with torch.no_grad():
    predicted_1 = net(speech_1.unsqueeze(0).to(device), contour_1.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    predicted_2 = net(speech_2.unsqueeze(0).to(device), contour_2.unsqueeze(0).to(device)).squeeze().cpu().numpy()


# # Example 1

speech_1 = speech_1.squeeze().numpy()
song_1 = song_1.squeeze().numpy()

draw_spectrograms(speech_1, song_1, 'Speech spectrogram (EX1)', 'Melody spectrogram (EX1)')
draw_single_spectrogram(predicted_1, 'Predicted spectrogram (EX1)')

real_total_energy_1, real_energy_per_frame_1 = compute_spectrogram_energy(song_1)
pred_total_energy_1, pred_energy_per_frame_1 = compute_spectrogram_energy(predicted_1)

print(f"Total energy (real): {real_total_energy_1}, Total energy (predicted): {pred_total_energy_1}")

plt.plot(real_energy_per_frame_1, label="Real", alpha=0.7)
plt.plot(pred_energy_per_frame_1, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Framewise spectrogram energy (EX1)")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.show()

# # Example 2

speech_2 = speech_2.squeeze().numpy()
song_2 = song_2.squeeze().numpy()

draw_spectrograms(speech_2, song_2, 'Speech spectrogram (EX2)', 'Melody spectrogram (EX2)')
draw_single_spectrogram(predicted_2, 'Predicted spectrogram (EX2)')

real_total_energy_2, real_energy_per_frame_2 = compute_spectrogram_energy(song_2)
pred_total_energy_2, pred_energy_per_frame_2 = compute_spectrogram_energy(predicted_2)

print(f"Total energy (real): {real_total_energy_2}, Total energy (predicted): {pred_total_energy_2}")

plt.plot(real_energy_per_frame_2, label="Real", alpha=0.7)
plt.plot(pred_energy_per_frame_2, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Framewise spectrogram energy (EX2)")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.show()

# Reconstruct
player = AudioPlayer()

# wavfile.write('../outputs/full/Example 7.wav', sr_1, audio_1)
# wavfile.write('../outputs/full/Example 8.wav', sr_2, audio_2)

original_audio_1, new_sr = player.play_audio_from_spectrogram(song_1, sr=16000, method='griffin-lim',
                                                     mode='return')
original_audio_2, _ = player.play_audio_from_spectrogram(song_2, sr=16000, method='griffin-lim',
                                                     mode='return')
predicted_audio_1, _ = player.play_audio_from_spectrogram(predicted_1, sr=16000, method='griffin-lim',
                                                     mode='return')
predicted_audio_2, _ = player.play_audio_from_spectrogram(predicted_2, sr=16000, method='griffin-lim',
                                                     mode='return')
path = PathRepo().get_test_wads_path()
original_audio_1 = (original_audio_1 * 32767).astype(np.int16)
original_audio_2 = (original_audio_2 * 32767).astype(np.int16)
predicted_audio_1 = (predicted_audio_1 * 32767).astype(np.int16)
predicted_audio_2 = (predicted_audio_2 * 32767).astype(np.int16)

wavfile.write(os.path.join(path, 'extest_1_original.wav'), new_sr, original_audio_1)
wavfile.write(os.path.join(path, 'extest_1_predicted.wav'), new_sr, predicted_audio_1)
wavfile.write(os.path.join(path, 'extest_2_original.wav'), new_sr, original_audio_2)
wavfile.write(os.path.join(path, 'extest_2_predicted.wav'), new_sr, predicted_audio_2)
