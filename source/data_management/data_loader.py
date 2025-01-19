
import os

import h5py
import librosa
import pandas as pd

from source.data_management.data_writer import DataWriter
from source.data_management.path_repo import PathRepo
from source.singleton import Singleton


class DataLoader(metaclass=Singleton):


    def load_raw_data(self, filename, load_wavs=False):
        if load_wavs:
            data = self.load_wavs()
            DataWriter().save_hdf5_raw(data, filename)
        else:
            data = self.load_raw_hdf5(filename)
        return data

    def load_preprocessed_data(self, filename):
        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index in f:
                contour = f[index]['contour'][:]
                melody_spectrogram = f[index]['melody_spectrogram'][:]
                speech_spectrogram = f[index]['speech_spectrogram'][:]
                melody_fs = f[index]['melody_fs'][:]
                speech_fs = f[index]['speech_fs'][:]
                data.append({'contour': contour, 'melody_spectrogram': melody_spectrogram,
                             'speech_spectrogram': speech_spectrogram, 'melody_fs': melody_fs, 'speech_fs': speech_fs})

        return pd.DataFrame(data)

    def load_wavs(self):

        wav_dir = PathRepo().get_wavs_path()
        data = []
        for subdir, dirs, files in os.walk(wav_dir):

            if 'sing' in dirs and 'read' in dirs:
                sing_path = os.path.join(subdir, 'sing')
                read_path = os.path.join(subdir, 'read')

                '''
                    Los archivos están nombrados con un número único, así que se listan por orden
                    y nos aseguramos de matchear los paired leidos y cantados
                '''
                for filename in sorted(os.listdir(sing_path)):
                    if filename.endswith('.wav'):
                        sing_file = os.path.join(sing_path, filename)
                        read_file = os.path.join(read_path, filename)

                        try:
                            sing_audio, _ = librosa.load(sing_file, sr=44100)
                            read_audio, _ = librosa.load(read_file, sr=44100)
                            data.append({'sing': sing_audio, 'read': read_audio})
                        except Exception as e:
                            print(f'Error loading {filename}: {e}')

        return pd.DataFrame(data)

    def load_raw_hdf5(self, filename='nus_train_raw.h5'):

        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index in f:
                sing_data = f[index]['sing'][:]
                read_data = f[index]['read'][:]
                data.append({'sing': sing_data, 'read': read_data})

        return pd.DataFrame(data)
