import numpy as np
import sounddevice as sd

from source.data_management.data_loader import DataLoader
from source.utilities import draw_spectrograms, draw_single_spectrogram

# Load test data
dataset = DataLoader.load_processed_data('nus_processed.h5', dataset=False)
speeches = dataset['speech']
temporal_dimensions = [speech.shape[-1] for speech in speeches]
print(f'Min = {min(temporal_dimensions)}')
print(f'Max = {max(temporal_dimensions)}')
print(f'Mean = {np.mean(temporal_dimensions)}')
print(f'Median = {np.median(temporal_dimensions)}')

for idx, row in dataset.iterrows():
    song = row['song']
    speech = row['speech']
    contour_spectrogram = row['contour']

    draw_spectrograms(speech, song, 'Spoken audio', 'Sang audio')
    draw_single_spectrogram(contour_spectrogram, 'Melody F0')
