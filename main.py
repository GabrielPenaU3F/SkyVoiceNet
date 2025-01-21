from source import utilities
from source.data_management.data_loader import DataLoader
from source.data_processing.preprocessing_pipeline import PreprocessingPipeline

# proc = PreprocessingPipeline()
#
# raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)
# processed_data = proc.preprocess(raw_data, save=True)

preprocessed_data = DataLoader.load_preprocessed_data('nus_processed.h5')

melody_spec = preprocessed_data['melody_spectrogram'].iloc[0]
speech_spec = preprocessed_data['speech_spectrogram'].iloc[0]
utilities.draw_spectrograms(melody_spec, speech_spec)
