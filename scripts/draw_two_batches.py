import torch
from torch.utils.data import DataLoader

from source.data_management.data_loader import DataLoader as custom_loader
from source.training_loop import collate_fn
from source.utilities import draw_spectrograms

dataset = custom_loader.load_processed_data('reduced_dataset_2.h5', dataset='variable')

torch.manual_seed(42)

# Train-test split
training_set, test_set = dataset.train_test_split(578, seed=42)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

for batch_idx, (speech_spec, melody_contour, melody_spec) in enumerate(dataloader):
    if batch_idx >= 2:
        break
    for i in range(speech_spec.shape[0]):
        speech_i = speech_spec[i, 0].cpu().numpy()
        melody_i = melody_spec[i, 0].cpu().numpy()
        draw_spectrograms(speech_i, melody_i, 'Speech', 'Melody')
