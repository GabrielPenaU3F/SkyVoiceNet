import os

import h5py
import numpy as np

from source.data_management.path_repo import PathRepo


class DataWriter:

    @staticmethod
    def save_hdf5_raw(data, filename, dtype='float32'):
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

    @staticmethod
    def save_hdf5_processed(data, filename, dtype='float32'):
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
                group.create_dataset('melody_sr', data=np.array(row['melody_sr'], dtype=dtype))
                group.create_dataset('speech_sr', data=np.array(row['speech_sr'], dtype=dtype))

        print('Data saved')
