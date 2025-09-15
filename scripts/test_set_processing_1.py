import os

import pandas as pd
import torch

from source.data_management.data_loader import DataLoader
from source.data_management.data_writer import DataWriter
from source.data_management.path_repo import PathRepo
from source.network.sky_voice_net import SkyVoiceNet
from source.utilities import match_spectrogram_dims

dataset = DataLoader.load_processed_data('test_set.h5', dataset='variable')
torch.manual_seed(42)
base_path = PathRepo().get_base_path()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_5_file = os.path.join(base_path, 'outputs', 'full', 'model_5_weights.pt')
model_7_file = os.path.join(base_path, 'outputs', 'full', 'model_7_weights.pt')

# Model 5
model_5 = SkyVoiceNet(dropout=0, reinforce_magnitude=4, residuals=None)
model_5_state_dict = torch.load(model_5_file, map_location=device)
model_5.load_state_dict(model_5_state_dict)
model_5.to(device)
model_5.eval()

# Model 7
model_7 = SkyVoiceNet(dropout=0.1, reinforce_magnitude=4, residuals=None)
model_7_state_dict = torch.load(model_7_file, map_location=device)
model_7.load_state_dict(model_7_state_dict)
model_7.to(device)
model_7.eval()

# Predicted spectrograms
model_5_predictions = []
model_7_predictions = []

for idx, datum in enumerate(dataset.data):
    speech = datum['speech']
    contour = datum['contour']

    min_len = min(speech.shape[-1], contour.shape[-1])
    speech = speech[..., :min_len]
    contour = contour[..., :min_len]

    speech_spectrogram_tensor = torch.tensor(speech).unsqueeze(0).unsqueeze(0).to('cuda').float()
    melody_contour_tensor = torch.tensor(contour).unsqueeze(0).unsqueeze(0).to('cuda').float()
    with torch.no_grad():
        model_5_pred = model_5(speech_spectrogram_tensor, melody_contour_tensor).squeeze()
        model_7_pred = model_5(speech_spectrogram_tensor, melody_contour_tensor).squeeze()
        model_5_predictions.append(model_5_pred.cpu().detach().numpy())
        model_7_predictions.append(model_7_pred.cpu().detach().numpy())

songs = [datum['song'] for datum in dataset.data]
test_data = pd.DataFrame({'original': songs, 'pred_5': model_5_predictions, 'pred_7': model_7_predictions})

for idx, row in test_data.iterrows():
    original = row['original']
    pred_5 = row['pred_5']
    pred_7 = row['pred_7']
    original, pred_5, pred_7 = match_spectrogram_dims(original, pred_5, pred_7)
    test_data.at[idx, 'original'] = original
    test_data.at[idx, 'pred_5'] = pred_5
    test_data.at[idx, 'pred_7'] = pred_7

DataWriter.save_test_data(test_data, 'test_pred_specs.h5')
