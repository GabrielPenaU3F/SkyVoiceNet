import os

import torch

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import draw_spectrograms, draw_single_spectrogram

net = SkyVoiceNet()

path_dir = PathRepo().get_output_path()
output_file = os.path.join(path_dir, 'sky_voice_net.pt')

trained_model = torch.load(output_file)

# Load test data
dataset = DataLoader.load_processed_data('reduced_dataset_2.h5', dataset=None)
sample = dataset.sample(n=1)
speech_spectrogram = sample.iloc[0]['speech']
melody_spectrogram = sample.iloc[0]['song']
melody_contour = sample.iloc[0]['contour']

# Shape [batch, channel, freq, time]
speech_spectrogram_tensor = torch.tensor(speech_spectrogram).unsqueeze(0).unsqueeze(0).to('cuda').float()
melody_contour_tensor = torch.tensor(melody_contour).unsqueeze(0).unsqueeze(0).to('cuda').float()

# Predict
with torch.no_grad():
    predicted_spectrogram = trained_model(speech_spectrogram_tensor, melody_contour_tensor).squeeze()

# Draw
predicted_spectrogram = predicted_spectrogram.cpu().numpy()
draw_spectrograms(speech_spectrogram, melody_spectrogram, 'Speech spectrogram', 'Melody spectrogram')
draw_single_spectrogram(predicted_spectrogram, 'Predicted')

# Reconstruct
player = AudioPlayer()
player.play_audio_from_spectrogram(predicted_spectrogram, sr=16000, compression_factor=2, denoising_strength=0.05)
