import os

import librosa

from source.data_management.path_repo import PathRepo
from source.data_processing.spectrogram_transformer import SpectrogramTransformer
from source.metric_calculator import MetricCalculator

metric_calc = MetricCalculator()
current = PathRepo().get_base_path()
base_path = os.path.split(current)[0]
path = os.path.join(base_path, 'SkyVoiceNet', 'test_audios')

melody_file = os.path.join(path, 'true_singing_0.wav')
parekh_predicted_file = os.path.join(path, 'parekh_predicted_singing_0.wav')
my_predicted_1_file = os.path.join(path, 'final_tests', 'predicted_model_1.wav')
my_predicted_2_file = os.path.join(path, 'final_tests', 'predicted_model_2.wav')
my_predicted_3_file = os.path.join(path, 'final_tests', 'predicted_model_3.wav')
my_predicted_4_file = os.path.join(path, 'final_tests', 'predicted_model_4.wav')
my_predicted_5_file = os.path.join(path, 'final_tests', 'predicted_model_5.wav')
my_predicted_6_file = os.path.join(path, 'final_tests', 'predicted_model_6.wav')
my_predicted_7_file = os.path.join(path, 'final_tests', 'predicted_model_7.wav')
my_predicted_8_file = os.path.join(path, 'final_tests', 'predicted_model_8.wav')

reference_melody, _ = librosa.load(melody_file, sr=16000)
parekh_predicted, _ = librosa.load(parekh_predicted_file, sr=16000)
my_predicted_1, _ = librosa.load(my_predicted_1_file, sr=16000)
my_predicted_2, _ = librosa.load(my_predicted_2_file, sr=16000)
my_predicted_3, _ = librosa.load(my_predicted_3_file, sr=16000)
my_predicted_4, _ = librosa.load(my_predicted_4_file, sr=16000)
my_predicted_5, _ = librosa.load(my_predicted_5_file, sr=16000)
my_predicted_6, _ = librosa.load(my_predicted_6_file, sr=16000)
my_predicted_7, _ = librosa.load(my_predicted_7_file, sr=16000)
my_predicted_8, _ = librosa.load(my_predicted_8_file, sr=16000)

# All audios are interpolated to the length of the original melody audio
n = len(reference_melody)
parekh_predicted = librosa.resample(parekh_predicted, orig_sr=16000, target_sr=16000 * (n/len(parekh_predicted)))
my_predicted_1 = librosa.resample(my_predicted_1, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_1)))
my_predicted_2 = librosa.resample(my_predicted_2, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_2)))
my_predicted_3 = librosa.resample(my_predicted_3, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_3)))
my_predicted_4 = librosa.resample(my_predicted_4, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_4)))
my_predicted_5 = librosa.resample(my_predicted_5, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_5)))
my_predicted_6 = librosa.resample(my_predicted_6, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_6)))
my_predicted_7 = librosa.resample(my_predicted_7, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_7)))
my_predicted_8 = librosa.resample(my_predicted_8, orig_sr=16000, target_sr=16000 * (n/len(my_predicted_8)))

# Spectrograms are obtained
ref_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    reference_melody, 1024, 256, 1024)
parekh_predicted_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    parekh_predicted, 1024, 256, 1024)
my_predicted_1_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_1, 1024, 256, 1024)
my_predicted_2_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_2, 1024, 256, 1024)
my_predicted_3_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_3, 1024, 256, 1024)
my_predicted_4_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_4, 1024, 256, 1024)
my_predicted_5_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_5, 1024, 256, 1024)
my_predicted_6_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_6, 1024, 256, 1024)
my_predicted_7_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_7, 1024, 256, 1024)
my_predicted_8_spectrogram = SpectrogramTransformer().obtain_log_spectrogram(
    my_predicted_8, 1024, 256, 1024)

# PSNR
psnr_parekh_ref = metric_calc.psnr(ref_spectrogram, parekh_predicted_spectrogram)
psnr_my_1_ref = metric_calc.psnr(ref_spectrogram, my_predicted_1_spectrogram)
psnr_my_2_ref = metric_calc.psnr(ref_spectrogram, my_predicted_2_spectrogram)
psnr_my_3_ref = metric_calc.psnr(ref_spectrogram, my_predicted_3_spectrogram)
psnr_my_4_ref = metric_calc.psnr(ref_spectrogram, my_predicted_4_spectrogram)
psnr_my_5_ref = metric_calc.psnr(ref_spectrogram, my_predicted_5_spectrogram)
psnr_my_6_ref = metric_calc.psnr(ref_spectrogram, my_predicted_6_spectrogram)
psnr_my_7_ref = metric_calc.psnr(ref_spectrogram, my_predicted_7_spectrogram)
psnr_my_8_ref = metric_calc.psnr(ref_spectrogram, my_predicted_8_spectrogram)

# PESQ
pesq_parekh_ref = metric_calc.pesq(reference_melody, parekh_predicted, 16000)
pesq_my_1_ref = metric_calc.pesq(reference_melody, my_predicted_1, 16000)
pesq_my_2_ref = metric_calc.pesq(reference_melody, my_predicted_2, 16000)
pesq_my_3_ref = metric_calc.pesq(reference_melody, my_predicted_3, 16000)
pesq_my_4_ref = metric_calc.pesq(reference_melody, my_predicted_4, 16000)
pesq_my_5_ref = metric_calc.pesq(reference_melody, my_predicted_5, 16000)
pesq_my_6_ref = metric_calc.pesq(reference_melody, my_predicted_6, 16000)
pesq_my_7_ref = metric_calc.pesq(reference_melody, my_predicted_7, 16000)
pesq_my_8_ref = metric_calc.pesq(reference_melody, my_predicted_8, 16000)

# SI-SDR
sisdr_parekh_ref = metric_calc.si_sdr(reference_melody, parekh_predicted)
sisdr_my_1_ref = metric_calc.si_sdr(reference_melody, my_predicted_1)
sisdr_my_2_ref = metric_calc.si_sdr(reference_melody, my_predicted_2)
sisdr_my_3_ref = metric_calc.si_sdr(reference_melody, my_predicted_3)
sisdr_my_4_ref = metric_calc.si_sdr(reference_melody, my_predicted_4)
sisdr_my_5_ref = metric_calc.si_sdr(reference_melody, my_predicted_5)
sisdr_my_6_ref = metric_calc.si_sdr(reference_melody, my_predicted_6)
sisdr_my_7_ref = metric_calc.si_sdr(reference_melody, my_predicted_7)
sisdr_my_8_ref = metric_calc.si_sdr(reference_melody, my_predicted_8)

# F0 RMSE
f0rmse_parekh_ref = metric_calc.f0_rmse(reference_melody, parekh_predicted)
f0rmse_my_1_ref = metric_calc.f0_rmse(reference_melody, my_predicted_1)
f0rmse_my_2_ref = metric_calc.f0_rmse(reference_melody, my_predicted_2)
f0rmse_my_3_ref = metric_calc.f0_rmse(reference_melody, my_predicted_3)
f0rmse_my_4_ref = metric_calc.f0_rmse(reference_melody, my_predicted_4)
f0rmse_my_5_ref = metric_calc.f0_rmse(reference_melody, my_predicted_5)
f0rmse_my_6_ref = metric_calc.f0_rmse(reference_melody, my_predicted_6)
f0rmse_my_7_ref = metric_calc.f0_rmse(reference_melody, my_predicted_7)
f0rmse_my_8_ref = metric_calc.f0_rmse(reference_melody, my_predicted_8)

# STOI
stoi_parekh_ref = metric_calc.stoi(reference_melody, parekh_predicted)
stoi_my_1_ref = metric_calc.stoi(reference_melody, my_predicted_1)
stoi_my_2_ref = metric_calc.stoi(reference_melody, my_predicted_2)
stoi_my_3_ref = metric_calc.stoi(reference_melody, my_predicted_3)
stoi_my_4_ref = metric_calc.stoi(reference_melody, my_predicted_4)
stoi_my_5_ref = metric_calc.stoi(reference_melody, my_predicted_5)
stoi_my_6_ref = metric_calc.stoi(reference_melody, my_predicted_6)
stoi_my_7_ref = metric_calc.stoi(reference_melody, my_predicted_7)
stoi_my_8_ref = metric_calc.stoi(reference_melody, my_predicted_8)

print('---- Parekh vs Reference ----')
print(f'PSNR = {psnr_parekh_ref:.2f} dB')
print(f'PESQ = {pesq_parekh_ref:.2f}')
print(f'SI-SDR = {sisdr_parekh_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_parekh_ref:.2f} Hz')
print(f'STOI = {stoi_parekh_ref:.2f}')
print('')

print('---- SkyVoiceNet Predictions ----')
print('')

print('-- Model 1 --')
print(f'PSNR = {psnr_my_1_ref:.2f} dB')
print(f'PESQ = {pesq_my_1_ref:.2f}')
print(f'SI-SDR = {sisdr_my_1_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_1_ref:.2f} Hz')
print(f'STOI = {stoi_my_1_ref:.2f}')
print('')
print('-- Model 2 --')
print(f'PSNR = {psnr_my_2_ref:.2f} dB')
print(f'PESQ = {pesq_my_2_ref:.2f}')
print(f'SI-SDR = {sisdr_my_2_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_2_ref:.2f} Hz')
print(f'STOI = {stoi_my_2_ref:.2f}')
print('')
print('-- Model 3 --')
print(f'PSNR = {psnr_my_3_ref:.2f} dB')
print(f'PESQ = {pesq_my_3_ref:.2f}')
print(f'SI-SDR = {sisdr_my_3_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_3_ref:.2f} Hz')
print(f'STOI = {stoi_my_3_ref:.2f}')
print('')
print('-- Model 4 --')
print(f'PSNR = {psnr_my_4_ref:.2f} dB')
print(f'PESQ = {pesq_my_4_ref:.2f}')
print(f'SI-SDR = {sisdr_my_4_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_4_ref:.2f} Hz')
print(f'STOI = {stoi_my_4_ref:.2f}')
print('')
print('-- Model 5 --')
print(f'PSNR = {psnr_my_5_ref:.2f} dB')
print(f'PESQ = {pesq_my_5_ref:.2f}')
print(f'SI-SDR = {sisdr_my_5_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_5_ref:.2f} Hz')
print(f'STOI = {stoi_my_5_ref:.2f}')
print('')
print('-- Model 6 --')
print(f'PSNR = {psnr_my_6_ref:.2f} dB')
print(f'PESQ = {pesq_my_6_ref:.2f}')
print(f'SI-SDR = {sisdr_my_6_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_6_ref:.2f} Hz')
print(f'STOI = {stoi_my_6_ref:.2f}')
print('')
print('-- Model 7 --')
print(f'PSNR = {psnr_my_7_ref:.2f} dB')
print(f'PESQ = {pesq_my_7_ref:.2f}')
print(f'SI-SDR = {sisdr_my_7_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_7_ref:.2f} Hz')
print(f'STOI = {stoi_my_7_ref:.2f}')
print('')
print('-- Model 8 --')
print(f'PSNR = {psnr_my_8_ref:.2f} dB')
print(f'PESQ = {pesq_my_8_ref:.2f}')
print(f'SI-SDR = {sisdr_my_8_ref:.2f} dB')
print(f'F0-RMSE = {f0rmse_my_8_ref:.2f} Hz')
print(f'STOI = {stoi_my_8_ref:.2f}')
