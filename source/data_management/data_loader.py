
import os

import h5py
import librosa
import numpy as np
import pandas as pd

from source.singleton import Singleton


class DataLoader(metaclass=Singleton):

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, filename, load_wavs=False):
        if load_wavs:
            data = self.load_wavs()
            self.save_hdf5(data, filename)
        else:
            data = self.load_hdf5(filename)
        return data

    def load_wavs(self):

        data = []
        for subdir, dirs, files in os.walk(self.data_dir):

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

    def save_hdf5(self, data, filename, dtype='float32'):
        with h5py.File(os.path.join(self.data_dir, filename), 'w') as f:

            for index, row in data.iterrows():
                group_name = str(index)
                if group_name in f:
                    del f[group_name]
                group = f.create_group(group_name)
                group.create_dataset('sing', data=np.array(row['sing'], dtype=dtype))
                group.create_dataset('read', data=np.array(row['read'], dtype=dtype))

        print("Data saved to HDF5 format.")

    def load_hdf5(self, filename='nus_train_raw.h5'):
        data = []
        with h5py.File(os.path.join(self.data_dir, filename), 'r') as f:
            for index in f:
                sing_data = f[index]['sing'][:]
                read_data = f[index]['read'][:]
                data.append({'sing': sing_data, 'read': read_data})

        return pd.DataFrame(data)