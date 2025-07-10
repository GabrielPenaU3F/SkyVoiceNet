import os

from source.singleton import Singleton


class PathRepo(metaclass=Singleton):

    def __init__(self):
        base_path = self.build_base_path()
        self.wavs_path = os.path.join(base_path, 'resources', 'NUS-48E', 'nus-smc-corpus_48')
        self.hdf5_path = os.path.join(base_path, 'resources', 'data')
        self.output_path = os.path.join(base_path, 'outputs')
        self.contour_path = os.path.join(base_path, 'resources', 'data', 'melody_contour')
        self.test_wavs_path = os.path.join(base_path, 'outputs', 'test_wavs')

    def get_wavs_path(self):
        return self.wavs_path

    def get_hdf5_path(self):
        return self.hdf5_path

    def build_base_path(self):
        current_path = os.getcwd()
        parent_dir = os.path.split(current_path)[0]
        return parent_dir

    def get_output_path(self):
        return self.output_path

    def get_contour_path(self):
        pass

    def get_test_wads_path(self):
        return self.test_wavs_path
