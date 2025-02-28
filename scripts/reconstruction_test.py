import os

import torch

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo

from source.network.sky_voice_net import SkyVoiceNet
from source.training_loop import train_model
from source.utilities import draw_spectrograms, draw_single_spectrogram

dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=True)
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

net = SkyVoiceNet(conv_out_channels=64)

# # Parameters
# batch_size = 4
# num_epochs = 10
# learning_rate = 1e-3
# loss_fn = torch.nn.MSELoss()
#
# # Train
# trained_model, training_loss = train_model(
#     model=net,
#     loss_fn=loss_fn,
#     dataset=dataset,
#     batch_size=batch_size,
#     num_epochs=num_epochs,
#     learning_rate=learning_rate,
#     device=device,
# )

path_dir = PathRepo().get_output_path()
output_file = os.path.join(path_dir, 'sky_voice_net.pt')
# torch.save(trained_model, output_file)

trained_model = torch.load(output_file)

# Load test data
dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=False)
speech_spectrogram = dataset.iloc[0]['speech_spectrogram']
melody_spectrogram = dataset.iloc[0]['melody_spectrogram']
melody_contour = dataset.iloc[0]['contour']
sr = dataset.iloc[0]['speech_sr']

# Shape [batch, channel, freq, time]
speech_spectrogram_tensor = torch.tensor(speech_spectrogram).unsqueeze(0).unsqueeze(0).to(device).float()
melody_contour_tensor = torch.tensor(melody_contour).unsqueeze(0).unsqueeze(0).to(device).float()

# Predict
with torch.no_grad():
    predicted_spectrogram = trained_model(speech_spectrogram_tensor, melody_contour_tensor).squeeze()

# Draw
predicted_spectrogram = predicted_spectrogram.cpu().numpy()
draw_spectrograms(melody_spectrogram, melody_contour, 'Melody spectrogram', 'Melody F0')
draw_single_spectrogram(predicted_spectrogram, 'Predicted spectrogram')

# Reconstruct
player = AudioPlayer()
player.play_audio_from_spectrogram(predicted_spectrogram, sr=sr, compression_factor=2, denoising_strength=0.05)
