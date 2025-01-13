import librosa
import numpy as np
import pandas as pd
from torchaudio.compliance.kaldi import spectrogram

from source.data_management.data_loader import DataLoader
from source.singleton import Singleton


class DataPreprocessor(metaclass=Singleton):

    def __init__(self):
        self.config = PreprocessConfig()

    def preprocess(self, data, **kwargs):
        self.config.update(**kwargs)  # Actualizar parámetros con kwargs
        processed_data = pd.DataFrame(columns=data.columns)
        for _, row in data.iterrows():
            norm_row = self.normalize_values(row)
            rr_row = self.resample(norm_row, self.config.target_fs)
            spectrogram_row = self.obtain_spectrograms(rr_row,
                                                       self.config.n_fft,
                                                       self.config.hop_length,
                                                       self.config.win_length)

            processed_data = pd.concat([processed_data, spectrogram_row])

        if self.config.save:
            self.save_processed_data(processed_data)

        return processed_data

    def resample(self, row, fs):
        sing = librosa.resample(row['sing'].values[0], orig_sr=44100, target_sr=fs)
        read = librosa.resample(row['read'].values[0], orig_sr=44100, target_sr=fs)
        return pd.DataFrame([{'sing': sing, 'read': read}])

    def obtain_spectrograms(self, row, n_fft, hop_length, win_length):
        sing_spectrogram = librosa.stft(row['sing'].values[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        read_spectrogram = librosa.stft(row['read'].values[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        sing_db = librosa.amplitude_to_db(abs(sing_spectrogram))
        read_db = librosa.amplitude_to_db(abs(read_spectrogram))
        return pd.DataFrame([{'sing': sing_db, 'read': read_db}])

    def normalize_values(self, row):
        sing = librosa.util.normalize(row['sing'])
        read = librosa.util.normalize(row['read'])
        return pd.DataFrame([{'sing': sing, 'read': read}])

    def save_processed_data(self, processed_data):
        DataLoader().save_hdf5(processed_data, filename='nus_train.h5')


class PreprocessConfig(metaclass=Singleton):

    def __init__(self):
        self.target_fs = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.save = False

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter: {key}')
