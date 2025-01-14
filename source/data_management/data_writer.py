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
                group_name = str(index)
                if group_name in f:
                    del f[group_name]
                group = f.create_group(group_name)
                group.create_dataset('sing', data=np.array(row['sing'], dtype=dtype))
                group.create_dataset('read', data=np.array(row['read'], dtype=dtype))

        print("Data saved to HDF5 format.")

    def save_hdf5_preprocessed(self, data, filename, dtype='float32'):
        pass
