from source.data_management.data_loader import DataLoader
from source.utilities import draw_spectrograms

dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=True)

speech_spec_tensor, contour_tensor, melody_spec_tensor, speech_sr, melody_sr = dataset[0]

print(speech_spec_tensor.shape)
print(contour_tensor.shape)
print(melody_spec_tensor.shape)
print(type(speech_spec_tensor))
print(type(melody_sr))
print(type(speech_sr))
