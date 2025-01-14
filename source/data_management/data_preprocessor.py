import librosa
import numpy as np
import pandas as pd

from source.data_management.data_writer import DataWriter
from source.singleton import Singleton


class DataPreprocessor(metaclass=Singleton):

    def __init__(self):
        self.config = PreprocessConfig()

    def preprocess(self, data, **kwargs):
        self.config.update(**kwargs)  # Actualizar parámetros con kwargs
        preprocessed_data = pd.DataFrame(columns=['contour', 'speech'])
        for _, row in data.iterrows():
            row_renamed = row.rename(columns=['contour', 'speech'])
            norm_row = self.normalize_values(row)
            silenceless_speech = self.remove_speech_silence(
                norm_row, self.config.fs, self.config.threshold, self.config.max_allowed_silence_duration)

            preprocessed_row = silenceless_speech
            preprocessed_data = pd.concat([preprocessed_data, preprocessed_row])

        if self.config.save:
            DataWriter().save_hdf5_preprocessed(preprocessed_data, self.config.filename)

        return preprocessed_data

    def remove_speech_silence(self, row, fs=44100, threshold=40, max_allowed_silence_duration=0.05):

        # Dividir audio en intervalos no silenciosos
        speech = row['read']
        non_silent_intervals = librosa.effects.split(speech, top_db=threshold)

        # Filtrar intervalos demasiado cortos (silencios menores a min_silence_duration)
        min_samples = int(max_allowed_silence_duration * fs)
        filtered_intervals = [i for i in non_silent_intervals if (i[1] - i[0]) >= min_samples]

        # Concatenar partes relevantes
        cleaned_audio = np.concatenate([speech[start:end] for start, end in filtered_intervals])

        return cleaned_audio

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


class PreprocessConfig(metaclass=Singleton):

    def __init__(self):
        self.fs = 44100
        self.threshold = 40
        self.max_allowed_silence_duration = 0.05
        self.target_fs = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.save = False
        self.filename = 'nus_processed.h5'

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter: {key}')
