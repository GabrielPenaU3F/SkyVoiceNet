from source.data_management.data_loader import DataLoader
from source.data_processing.processing_pipeline import ProcessingPipeline

proc = ProcessingPipeline()

raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)
processed_data = proc.process(raw_data, save=True, normalize=False, filename='nus_processed_no_norm.h5')
