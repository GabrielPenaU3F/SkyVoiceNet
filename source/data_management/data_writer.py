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
                group.create_dataset('marks_sing', data=row['marks_sing'].to_json())
                group.create_dataset('marks_read', data=row['marks_read'].to_json())

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
                group.create_dataset('song', data=np.array(row['song'], dtype=dtype))
                group.create_dataset('speech', data=np.array(row['speech'], dtype=dtype))

        print('Data saved')

    @staticmethod
    def save_test_data(data, filename, dtype='float32'):
        filepath = os.path.join(PathRepo().get_hdf5_path(), filename)
        with h5py.File(filepath, 'w') as f:

            for index, row in data.iterrows():
                print(f'Saving input element Nº {index + 1}')
                group_name = str(index)
                if group_name in f:
                    del f[group_name]
                group = f.create_group(group_name)
                group.create_dataset('original', data=np.array(row['original'], dtype=dtype))
                group.create_dataset('pred_5', data=np.array(row['pred_5'], dtype=dtype))
                group.create_dataset('pred_7', data=np.array(row['pred_7'], dtype=dtype))

        print('Data saved')