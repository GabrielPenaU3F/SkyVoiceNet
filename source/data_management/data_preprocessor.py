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
        data = data.rename(columns={'sing': 'contour', 'read': 'speech'})
        for _, row in data.iterrows():
            norm_row = self.normalize_values(row)
            silenceless_speech = self.remove_speech_silence(
                norm_row, self.config.fs, self.config.threshold, self.config.max_allowed_silence_duration)

            preprocessed_row = silenceless_speech
            preprocessed_data = pd.concat([preprocessed_data, preprocessed_row])

        if self.config.save:
            DataWriter().save_hdf5_preprocessed(preprocessed_data, self.config.filename)

        return preprocessed_data

    def remove_speech_silence(self, row, fs=44100, threshold=40, max_allowed_silence_duration=0.05):

        sing = row['contour'].values[0]
        speech = row['speech'].values[0]
        # Divide audio into non-silent intervals
        non_silent_intervals = librosa.effects.split(speech, top_db=threshold)
        # By default, this covers 50ms
        max_silence_samples = int(max_allowed_silence_duration * fs)
        silent_intervals = self.obtain_silent_intervals(speech, non_silent_intervals)
        # This filters the array to identify long silences
        filtered_silent_intervals = [
            (start, end) for start, end in silent_intervals if (end - start) >= max_silence_samples
        ]

        # Finally, we rebuild the audio omitting the long silences
        filtered_fragments = self.rebuild_audio_without_long_silences(speech, filtered_silent_intervals)
        clean_speech = np.concatenate(filtered_fragments)

        return pd.DataFrame([{'contour': sing, 'speech': clean_speech}])

    def rebuild_audio_without_long_silences(self, speech, filtered_silent_intervals):
        filtered_audio = []
        prev_end = 0
        for start, end in filtered_silent_intervals:
            filtered_audio.append(speech[prev_end:start])
            prev_end = end
        if prev_end < len(speech):
            filtered_audio.append(speech[prev_end:])
        return filtered_audio

    def obtain_silent_intervals(self, speech, non_silent_intervals):
        silent_intervals = []
        start_index = 0
        for interval in non_silent_intervals:
            # Add the next silent interval
            if start_index < interval[0]:
                silent_intervals.append([start_index, interval[0] - 1])

            start_index = interval[1] + 1
        # Add the last silent interval, if it exists
        if start_index < len(speech):
            silent_intervals.append([start_index, len(speech) - 1])

        return silent_intervals

    def obtain_spectrograms(self, row, n_fft, hop_length, win_length):
        sing_spectrogram = librosa.stft(row['sing'].values[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        read_spectrogram = librosa.stft(row['read'].values[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        sing_db = librosa.amplitude_to_db(abs(sing_spectrogram))
        read_db = librosa.amplitude_to_db(abs(read_spectrogram))
        return pd.DataFrame([{'sing': sing_db, 'read': read_db}])

    def normalize_values(self, row):
        sing = librosa.util.normalize(row['contour'])
        read = librosa.util.normalize(row['speech'])
        return pd.DataFrame([{'contour': sing, 'speech': read}])


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
