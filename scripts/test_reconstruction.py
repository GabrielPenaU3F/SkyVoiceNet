from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader

dataset = DataLoader.load_processed_data('nus_processed_no_norm.h5', as_tensor_dataset=False)

spec = dataset.iloc[0]['melody_spectrogram']
spec_sr = dataset.iloc[0]['melody_sr']

player = AudioPlayer()
player.play_audio_from_spectrogram(spec, sr=spec_sr, compression_factor=3, denoising_strength=0.05)
