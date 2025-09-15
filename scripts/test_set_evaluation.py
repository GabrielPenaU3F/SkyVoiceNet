import os

import pandas as pd
import torch

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.metric_calculator import MetricCalculator

test_specs = DataLoader.load_test_data('test_pred_specs.h5')
test_audios = DataLoader.load_test_data('test_pred_audios.h5')
calc = MetricCalculator()
torch.manual_seed(42)

psnrs_5 = []
psnrs_7 = []
pesqs_5 = []
pesqs_7 = []
stois_5 = []
stois_7 = []
f0_rmses_5 = []
f0_rmses_7 = []

for idx, (spec_datum, audio_datum) in enumerate(zip(test_specs.data, test_audios.data)):

    print(f'Processing audio Nº{idx + 1}')

    original_spec = spec_datum['original']
    original_audio = audio_datum['original']
    pred_5_spec = spec_datum['pred_5']
    pred_5_audio = audio_datum['pred_5']
    pred_7_spec = spec_datum['pred_7']
    pred_7_audio = audio_datum['pred_7']

    psnr_5 = calc.psnr(original_spec, pred_5_spec)
    psnr_7 = calc.psnr(original_spec, pred_7_spec)
    psnrs_5.append(psnr_5)
    psnrs_7.append(psnr_7)

    pesq_5 = calc.pesq(original_audio, pred_5_audio)
    pesq_7 = calc.pesq(original_audio, pred_7_audio)
    pesqs_5.append(pesq_5)
    pesqs_7.append(pesq_7)

    stoi_5 = calc.stoi(original_audio, pred_5_audio)
    stoi_7 = calc.stoi(original_audio, pred_7_audio)
    stois_5.append(stoi_5)
    stois_7.append(stoi_7)

    f0_rmse_5 = calc.f0_rmse(original_audio, pred_5_audio)
    f0_rmse_7 = calc.f0_rmse(original_audio, pred_7_audio)
    f0_rmses_5.append(f0_rmse_5)
    f0_rmses_7.append(f0_rmse_7)

metrics_5 = {
    "PSNR (dB)": psnrs_5,
    "PESQ": pesqs_5,
    "STOI": stois_5,
    "F0 RMSE (Hz)": f0_rmses_5,
}

metrics_7 = {
    "PSNR (dB)": psnrs_7,
    "PESQ": pesqs_7,
    "STOI": stois_7,
    "F0 RMSE (Hz)": f0_rmses_7,
}

df5 = pd.DataFrame(metrics_5)
df5["model"] = "model_5"

df7 = pd.DataFrame(metrics_7)
df7["model"] = "model_7"

filepath = os.path.join(PathRepo().get_hdf5_path(), 'metrics_all.h5')
df_all = pd.concat([df5, df7], ignore_index=True)
df_all.to_hdf(filepath, key="metrics", mode="w")
