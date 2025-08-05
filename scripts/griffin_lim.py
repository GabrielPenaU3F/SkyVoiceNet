import os

import librosa
import sounddevice as sd

from source.audio_reconstruction.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.data_processing.resampler import Resampler
from source.data_processing.spectrogram_transformer import SpectrogramTransformer

transformer = SpectrogramTransformer()
wav_dir = PathRepo().get_wavs_path()
sing_file = os.path.join(wav_dir, 'ADIZ', 'sing', '01.wav')
sing_audio, _ = librosa.load(sing_file, sr=44100)
# sd.play(sing_audio, 44100)
# sd.wait()

print("Audio duration: ", len(sing_audio)/44100)
resampled_audio = Resampler(44100).resample(sing_audio, 16000)
log_spectrogram = transformer.obtain_log_spectrogram(resampled_audio, n_fft=1024, hop_length=256, win_length=1024)

# Reconstruct

print("Spectrogram shape:", log_spectrogram.shape)
print("Estimated duration:", log_spectrogram.shape[1] * 256 / 16000, "seconds")

player = AudioPlayer()
player.play_audio_from_spectrogram(
    log_spectrogram, sr=16000, method='griffin-lim', mode='play')
