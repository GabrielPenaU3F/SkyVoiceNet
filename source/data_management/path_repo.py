from source.singleton import Singleton


class PathRepo(metaclass=Singleton):

    def __init__(self):
        self.wavs_path = 'resources/NUS-48E/nus-smc-corpus_48'
        self.hdf5_path = 'resources/data'

    def get_wavs_path(self):
        return self.wavs_path

    def get_hdf5_path(self):
        return self.hdf5_path
