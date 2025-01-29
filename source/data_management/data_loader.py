
import os

import h5py
import librosa
import pandas as pd

from source.data_management.data_writer import DataWriter
from source.data_management.path_repo import PathRepo

class DataLoader:

    @staticmethod
    def load_raw_data(filename, load_wavs=False):
        if load_wavs:
            data = DataLoader.load_wavs()
            DataWriter.save_hdf5_raw(data, filename)
        else:
            data = DataLoader.load_raw_hdf5(filename)
        return data

    @staticmethod
    def load_processed_data(filename):
        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index in f:
                contour = f[index]['contour'][:]
                melody_spectrogram = f[index]['melody_spectrogram'][:]
                speech_spectrogram = f[index]['speech_spectrogram'][:]
                melody_sr = f[index]['melody_sr'][()]
                speech_sr = f[index]['speech_sr'][()]
                data.append({'contour': contour, 'melody_spectrogram': melody_spectrogram,
                             'speech_spectrogram': speech_spectrogram, 'melody_sr': melody_sr, 'speech_sr': speech_sr})

        return pd.DataFrame(data)

    @staticmethod
    def load_wavs():

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

    def load_raw_hdf5(self, filename='nus_data_raw.h5'):

        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index in f:
                sing_data = f[index]['sing'][:]
                read_data = f[index]['read'][:]
                data.append({'sing': sing_data, 'read': read_data})

        return pd.DataFrame(data)
