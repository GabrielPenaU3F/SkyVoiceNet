from source.data_management.data_loader import DataLoader

dataset = DataLoader.load_processed_data('nus_processed.h5', as_tensor_dataset=True)

speech_spec_tensor, contour_tensor, melody_spec_tensor = dataset[0]

print(speech_spec_tensor.shape)
print(contour_tensor.shape)
print(melody_spec_tensor.shape)
print(type(speech_spec_tensor))
