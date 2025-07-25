from source.data_management.data_loader import DataLoader
from source.data_processing.processing_pipeline import ProcessingPipeline

proc = ProcessingPipeline()

# raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)
# processed_data = proc.process(raw_data, mode='pre', save=True, filename='nus_preprocessed_3.h5')

raw_data = DataLoader.load_preprocessed_data('nus_preprocessed_3.h5')
processed_data = proc.process(raw_data, mode='post', save=True, filename='nus_processed_3.h5')
