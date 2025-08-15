import os

import librosa
import numpy as np
import torch
from scipy.io import wavfile

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.path_repo import PathRepo
from source.data_processing.contour_extractor import ContourExtractor
from source.data_processing.spectrogram_transformer import SpectrogramTransformer
from source.network.min_max_wrapper import MinMaxWrapper
from source.utilities import draw_spectrograms, draw_single_spectrogram

base_path = PathRepo().get_base_path()
path = os.path.join(base_path, 'resources', 'parekh')

speech_file = os.path.join(path, 'stretched_speech_0.wav')
melody_file = os.path.join(path, 'true_singing_0.wav')
speech, _ = librosa.load(speech_file, sr=16000)
melody, _ = librosa.load(melody_file, sr=16000)

speech_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    speech, 1024, 256, 1024)

contour = ContourExtractor().extract_contour(melody, sr=16000)

draw_spectrograms(speech_spectrogram, contour, 'Speech spectrogram', 'Contour')

speech_spectrogram_tensor = torch.tensor(speech_spectrogram).unsqueeze(0).unsqueeze(0).to('cuda').float()
melody_contour_tensor = torch.tensor(contour).unsqueeze(0).unsqueeze(0).to('cuda').float()

# Network

model_file = os.path.join(base_path, 'outputs', 'full', 'sky_voice_net_4_full_batch4_test_7.pt')
trained_model = torch.load(model_file)
net = MinMaxWrapper(trained_model)
net.eval()
with torch.no_grad():
    predicted_spectrogram = net(speech_spectrogram_tensor, melody_contour_tensor).squeeze()

draw_single_spectrogram(predicted_spectrogram.cpu().numpy(), 'Predicted')

# Audio

player = AudioPlayer()
predicted_audio, _ = player.play_audio_from_spectrogram(predicted_spectrogram,
                                                        sr=16000, method='griffin-lim', mode='return')
predicted_audio = (predicted_audio * 32767).astype(np.int16)
wavfile.write(os.path.join(path, 'predicted_test7.wav'), 16000, predicted_audio)
