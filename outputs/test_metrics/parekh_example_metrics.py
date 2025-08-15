import os

import librosa

from source.data_management.path_repo import PathRepo
from source.data_processing.spectrogram_transformer import SpectrogramTransformer
from source.utilities import psnr, audio_pesq

current = PathRepo().get_base_path()
base_path = os.path.split(current)[0]
path = os.path.join(base_path, 'resources', 'parekh')

melody_file = os.path.join(path, 'true_singing_0.wav')
parekh_predicted_file = os.path.join(path, 'predicted_singing_0.wav')
my_predicted_file = os.path.join(path, 'predicted_test7.wav')
reference_melody, _ = librosa.load(melody_file, sr=16000)
parekh_predicted, _ = librosa.load(parekh_predicted_file, sr=16000)
my_predicted, _ = librosa.load(my_predicted_file, sr=16000)

ref_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    reference_melody, 1024, 256, 1024)
parekh_predicted_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    parekh_predicted, 1024, 256, 1024)
my_predicted_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted, 1024, 256, 1024)

# El espectrograma del audio reconstruido de Parekh tiene un frame extra... (raro)
min_frames = min(ref_spectrogram.shape[1], parekh_predicted_spectrogram.shape[1]) # 151 frames
parekh_predicted_spectrogram = parekh_predicted_spectrogram[:, :min_frames]

psnr_parekh_ref = psnr(ref_spectrogram, parekh_predicted_spectrogram)
psnr_my_ref = psnr(ref_spectrogram, my_predicted_spectrogram)
pesq_parekh_ref = audio_pesq(reference_melody, parekh_predicted, 16000)
pesq_my_ref = audio_pesq(reference_melody, my_predicted, 16000)

print('---- Parekh vs Reference ----')
print(f'PSNR = {psnr_parekh_ref:.2f} dB')
print(f'PESQ = {pesq_parekh_ref:.2f}')
print('')

print('---- My Audio vs Reference ----')
print(f'PSNR = {psnr_my_ref:.2f} dB')
print(f'PESQ = {pesq_my_ref:.2f}')
