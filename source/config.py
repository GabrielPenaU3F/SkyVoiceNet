import torch

from source.singleton import Singleton

class Config(metaclass=Singleton):

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter: {key}')

class PreprocessConfig(Config):

    def __init__(self):
        self.original_sr = 44100
        self.resample_sr = 16000
        self.silence_threshold = 40
        self.max_allowed_silence_duration = 0.05
        self.target_sr = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.crepe_fmin = 50
        self.crepe_fmax = 1500
        self.crepe_batch_size = 2048
        self.crepe_filter_win_frames = 3
        self.min_resample_factor = 0.9
        self.max_resample_factor = 1.1
        self.save = False
        self.filename = 'nus_processed.h5'


class CrepeConfig(Config):

    def __init__(self):
        global_config = PreprocessConfig()
        self.silence_threshold = global_config.silence_threshold
        self.hop_length = global_config.hop_length
        self.fmin = global_config.crepe_fmin
        self.fmax = global_config.crepe_fmax
        self.batch_size = global_config.crepe_batch_size
        self.filter_win_frames = global_config.crepe_filter_win_frames

    def expand(self):
        return self.silence_threshold, self.hop_length, self.fmin, self.fmax, self.batch_size, self.filter_win_frames


class NetworkConfig(Config):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv_out_channels = 64
        self.transf_dim = 256
        self.transf_heads = 4
        self.transf_hidden = 512
        self.transf_num_layers = 6
        self.transf_dropout = 0.1
        self.cross_attention_num_heads = 8
        self.cross_attention_dropout = 0.1
        self.conv_output_dim = None


class AudioPlayerConfig(Config):

    def __init__(self):
        self.device = 'cuda'
        self.sr = 22050
        self.gan_sr = 22050
        self.n_fft = 1024
        self.n_mels = 80
        self.compression_factor = 1
        self.denoising_strength = 0.005
