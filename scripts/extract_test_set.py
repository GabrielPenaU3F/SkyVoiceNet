import pandas as pd
import torch

from source.data_management.data_loader import DataLoader
from source.data_management.data_writer import DataWriter

dataset = DataLoader.load_processed_data('nus_processed_7.h5', dataset='variable')
torch.manual_seed(42)

# Train-test split
_, test_set = dataset.train_test_split(578, seed=42)

test_df = pd.DataFrame(test_set.data)
print(test_df.head())

DataWriter().save_hdf5_processed(test_df, 'test_set.h5')
