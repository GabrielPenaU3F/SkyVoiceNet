import os

import h5py
import numpy as np

from source.data_management.path_repo import PathRepo
from source.singleton import Singleton


class DataWriter(metaclass=Singleton):

    def save_hdf5_raw(self, data, filename, dtype='float32'):
        filepath = os.path.join(PathRepo().get_hdf5_path(), filename)
        with h5py.File(filepath, 'w') as f:

            for index, row in data.iterrows():
                print(f'Saving song Nº {index + 1}')
                group_name = str(index)
                if group_name in f:
                    del f[group_name]
                group = f.create_group(group_name)
                group.create_dataset('sing', data=np.array(row['sing'], dtype=dtype))
                group.create_dataset('read', data=np.array(row['read'], dtype=dtype))

        print('Data saved')

    def save_hdf5_preprocessed(self, data, filename, dtype='float32'):
        filepath = os.path.join(PathRepo().get_hdf5_path(), filename)
        with h5py.File(filepath, 'w') as f:

            for index, row in data.iterrows():
                print(f'Saving input element Nº {index + 1}')
                group_name = str(index)
                if group_name in f:
                    del f[group_name]
                group = f.create_group(group_name)
                group.create_dataset('contour', data=np.array(row['contour'], dtype=dtype))
                group.create_dataset('melody_spectrogram', data=np.array(row['melody_spectrogram'], dtype=dtype))
                group.create_dataset('speech_spectrogram', data=np.array(row['speech_spectrogram'], dtype=dtype))
                group.create_dataset('melody_fs', data=np.array(row['melody_fs'], dtype=dtype))
                group.create_dataset('speech_fs', data=np.array(row['speech_fs'], dtype=dtype))

        print('Data saved')
