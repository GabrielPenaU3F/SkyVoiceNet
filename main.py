from source.data_management.data_loader import DataLoader
from source.data_management.data_preprocessor import DataPreprocessor

data = DataLoader('resources/NUS-48E/nus-smc-corpus_48').load_data(force_reload=False)
processed_data = DataPreprocessor().preprocess(data, save=False)

