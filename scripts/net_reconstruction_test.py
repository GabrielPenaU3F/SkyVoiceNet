import os

import torch

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.utilities import draw_spectrograms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=False)

# Load pretrained model
outputs_path = PathRepo().get_output_path()
path = os.path.join(outputs_path, "sky_voice_net.pt")
model = torch.load(path)
model.eval()

# Load test data
speech_spectrogram = dataset.iloc[0]['speech_spectrogram']
melody_spectrogram = dataset.iloc[0]['melody_spectrogram']
melody_contour = dataset.iloc[0]['contour']
sr = dataset.iloc[0]['speech_sr']

# Shape [batch, channel, freq, time]
speech_spectrogram = torch.tensor(speech_spectrogram).unsqueeze(0).unsqueeze(0).to(device).float()
melody_contour = torch.tensor(melody_contour).unsqueeze(0).unsqueeze(0).to(device).float()

# Predict
with torch.no_grad():
    predicted_spectrogram = model(speech_spectrogram, melody_contour).squeeze()

# Draw
predicted_spectrogram = predicted_spectrogram.cpu().numpy()
draw_spectrograms(melody_spectrogram, predicted_spectrogram)

# Reconstruct
player = AudioPlayer()
player.play_audio_from_spectrogram(predicted_spectrogram, sr=sr, compression_factor=2, denoising_strength=0.05)
