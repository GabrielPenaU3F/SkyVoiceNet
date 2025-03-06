import numpy as np

from source.data_management.data_loader import DataLoader
from source.utilities import draw_spectrograms, draw_single_spectrogram

dataset = DataLoader.load_processed_data('nus_processed_nopad.h5', dataset=False)

# Load test data
dataset = DataLoader.load_processed_data('nus_processed_nopad.h5',dataset=False)
speech_spectrogram = dataset.iloc[0]['speech_spectrogram']
melody_spectrogram = dataset.iloc[0]['melody_spectrogram']
melody_contour = dataset.iloc[0]['contour']
sr = dataset.iloc[0]['speech_sr']


draw_spectrograms(speech_spectrogram, melody_contour, 'Speech spectrogram', 'Melody F0')
draw_single_spectrogram(melody_spectrogram, 'Melody spectrogram')
