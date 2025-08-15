from source.data_management.data_loader import DataLoader
from source.data_management.data_writer import DataWriter

dataset = DataLoader.load_processed_data('nus_processed_5.h5', dataset=False)

reduced_dataset = dataset[:1600]
DataWriter.save_hdf5_processed(reduced_dataset, 'reduced_dataset_5.h5')
