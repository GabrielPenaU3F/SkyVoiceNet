import os

import librosa
import numpy as np
import torch
from scipy.io import wavfile

from source.audio_player import AudioPlayer
from source.data_management.path_repo import PathRepo
from source.data_processing.contour_extractor import ContourExtractor
from source.data_processing.spectrogram_transformer import SpectrogramTransformer
from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import draw_spectrograms, draw_single_spectrogram, amplify_audio

base_path = PathRepo().get_base_path()
path = os.path.join(base_path, 'test_audios')

speech_file = os.path.join(path, 'stretched_speech_0.wav')
melody_file = os.path.join(path, 'true_singing_0.wav')
speech, _ = librosa.load(speech_file, sr=16000)
melody, _ = librosa.load(melody_file, sr=16000)

speech_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    speech, 1024, 256, 1024, keep_last_freq=False)
contour = ContourExtractor().extract_contour(melody, sr=16000)

draw_spectrograms(speech_spectrogram, contour, 'Speech spectrogram', 'Contour')

speech_spectrogram_tensor = torch.tensor(speech_spectrogram).unsqueeze(0).unsqueeze(0).to('cuda').float()
melody_contour_tensor = torch.tensor(contour).unsqueeze(0).unsqueeze(0).to('cuda').float()

# Network

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_file = os.path.join(base_path, 'outputs', 'full', 'model_8_weights.pt')
model = SkyVoiceNet(dropout=0,
                    reinforce_magnitude=4,
                    residuals=None
                    )
trained_state_dict = torch.load(model_file, map_location=device)
model.load_state_dict(trained_state_dict)
model.to(device)
model.eval()
with torch.no_grad():
    predicted_spectrogram = model(speech_spectrogram_tensor, melody_contour_tensor).squeeze()

predicted_spectrogram = predicted_spectrogram.cpu().numpy()
draw_single_spectrogram(predicted_spectrogram, 'Predicted')

# Audio

player = AudioPlayer()
predicted_audio, _ = player.regenerate_audio(predicted_spectrogram, sr=16000, mode='return', amplify=True)
predicted_audio = (predicted_audio * 32767).astype(np.int16)
wavfile.write(os.path.join(path, 'final_tests', 'predicted_model_8.wav'), 16000, predicted_audio)
