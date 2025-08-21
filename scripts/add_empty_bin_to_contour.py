import numpy as np

from source.data_management.data_loader import DataLoader
from source.data_management.data_writer import DataWriter

dataset = DataLoader.load_processed_data('nus_processed_6.h5', dataset=False)

for idx, contour in dataset['contour'].items():
    T = contour.shape[1]
    zeroes = np.zeros((1, T))
    dataset.at[idx, 'contour'] = np.vstack([contour, zeroes])

DataWriter().save_hdf5_processed(dataset, 'nus_processed_6.h5')
