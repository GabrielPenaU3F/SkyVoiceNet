
import os
from io import StringIO

import h5py
import librosa
import pandas as pd
from torch.utils.data import TensorDataset

from source.data_management.data_writer import DataWriter
from source.data_management.path_repo import PathRepo
from source.data_management.variable_dim_dataset import VariableLengthDataset
from source.utilities import series_to_tensor


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
    def load_processed_data(filename, dataset=None):
        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index in f:
                contour = f[index]['contour'][:]
                song = f[index]['song'][:]
                speech = f[index]['speech'][:]
                data.append({'contour': contour, 'song': song, 'speech': speech})

        data = pd.DataFrame(data)
        if dataset == 'tensor':
            speech_spec = series_to_tensor(data['speech'])
            contour_spec = series_to_tensor(data['contour'])
            target_spec = series_to_tensor(data['song'])
            data = TensorDataset(speech_spec, contour_spec, target_spec)
        elif dataset == 'variable':
            data_list = data.apply(lambda row: {
                'speech': row['speech'],
                'contour': row['contour'],
                'song': row['song']
            }, axis=1).tolist()
            data = VariableLengthDataset(data_list)
        return data

    @staticmethod
    def load_test_data(filename):
        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index in f:
                original = f[index]['original'][:]
                pred_5 = f[index]['pred_5'][:]
                pred_7 = f[index]['pred_7'][:]
                data.append({'original': original, 'pred_5': pred_5, 'pred_7': pred_7})

        data = pd.DataFrame(data)
        data_list = data.apply(lambda row: {
            'original': row['original'],
            'pred_5': row['pred_5'],
            'pred_7': row['pred_7']
        }, axis=1).tolist()
        data = VariableLengthDataset(data_list)

        return data

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
                filenames = sorted(f for f in os.listdir(sing_path) if f.endswith('.wav'))
                for filename in filenames:

                    sing_file = os.path.join(sing_path, filename)
                    read_file = os.path.join(read_path, filename)
                    sing_marks_file = os.path.join(sing_path, filename.replace('.wav', '.txt'))
                    read_marks_file = os.path.join(read_path, filename.replace('.wav', '.txt'))

                    sing_audio, read_audio = None, None
                    sing_marks, read_marks = None, None

                    try:
                        sing_audio, _ = librosa.load(sing_file, sr=44100)
                        read_audio, _ = librosa.load(read_file, sr=44100)
                        sing_marks = DataLoader.load_marks(filename=sing_marks_file)
                        read_marks = DataLoader.load_marks(filename=read_marks_file)

                    except Exception as e:
                        print(f'Error loading {filename}: {e}')


                    data.append({'sing': sing_audio, 'read': read_audio,
                                 'marks_sing': sing_marks, 'marks_read': read_marks})

        return pd.DataFrame(data)

    @staticmethod
    def load_raw_hdf5(filename='nus_data_raw.h5'):

        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index, row in f.items():
                sing_data = row['sing'][:]
                read_data = row['read'][:]
                marks_sing = pd.read_json(StringIO(row['marks_sing'][()].decode('utf-8')))
                marks_read = pd.read_json(StringIO(row['marks_read'][()].decode('utf-8')))
                data.append({'sing': sing_data, 'read': read_data, 'marks_sing': marks_sing, 'marks_read': marks_read})

        return pd.DataFrame(data)

    @staticmethod
    def load_marks(filename):
        # Load textfiles into a dataframe
        df = pd.read_csv(filename, sep=r'\s+', names=['start', 'end', 'mark'],
                         dtype={'start': float, 'end': float, 'mark': str})
        return df

    @staticmethod
    def load_preprocessed_data(filename):
        hdf5_dir = PathRepo().get_hdf5_path()
        data = []
        with h5py.File(os.path.join(hdf5_dir, filename), 'r') as f:
            for index, row in f.items():
                speech = row['speech'][:]
                contour = row['contour'][:]
                song = row['song'][:]

                data.append({'song': song, 'speech': speech, 'contour': contour})

        return pd.DataFrame(data)
