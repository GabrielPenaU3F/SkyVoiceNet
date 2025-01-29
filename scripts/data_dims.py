from source.data_management.data_loader import DataLoader
from source.utilities import draw_spectrograms

processed_data = DataLoader.load_processed_data('nus_processed.h5')

contour_spec = processed_data['contour'].values
speech_spec = processed_data['speech_spectrogram'].values
melody_spec = processed_data['melody_spectrogram'].values

print([spec.shape for spec in contour_spec])
print([spec.shape for spec in speech_spec])
print([spec.shape for spec in melody_spec])

draw_spectrograms(contour_spec[0], speech_spec[0])
