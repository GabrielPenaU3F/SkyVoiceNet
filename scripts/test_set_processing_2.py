import pandas as pd
import torch

from source.audio_player import AudioPlayer
from source.data_management.data_loader import DataLoader
from source.data_management.data_writer import DataWriter

dataset = DataLoader.load_test_data('test_pred_specs.h5')
player = AudioPlayer()
torch.manual_seed(42)

original_audios = []
pred_5_audios = []
pred_7_audios = []

for idx, datum in enumerate(dataset.data):
    original_spec = datum['original']
    pred_5 = datum['pred_5']
    pred_7 = datum['pred_7']

    original_audio, _ = player.regenerate_audio(original_spec, sr=16000, mode='return', amplify=True)
    pred_5_audio, _ = player.regenerate_audio(pred_5, sr=16000, mode='return', amplify=True)
    pred_7_audio, _ = player.regenerate_audio(pred_7, sr=16000, mode='return', amplify=True)

    original_audios.append(original_audio)
    pred_5_audios.append(pred_5_audio)
    pred_7_audios.append(pred_7_audio)

test_data = pd.DataFrame({'original': original_audios, 'pred_5': pred_5_audios, 'pred_7': pred_7_audios})
DataWriter.save_test_data(test_data, 'test_pred_audios.h5')
