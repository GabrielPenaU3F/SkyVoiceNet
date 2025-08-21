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
        self.mode = 'full'
        self.original_sr = 44100
        self.resample_sr = 16000
        self.silence_threshold = 40
        self.max_allowed_silence_duration = 0.05
        self.target_sr = 16000
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.keep_last_freq = False
        self.crepe_fmin = 50
        self.crepe_fmax = 1500
        self.crepe_batch_size = 2048
        self.crepe_filter_win_frames = 3
        self.min_resample_factor = 0.9
        self.max_resample_factor = 1.1
        self.save = False
        self.filename = 'nus_processed.h5'
        self.padding = 'zero'


class CrepeConfig(Config):

    def __init__(self):
        global_config = PreprocessConfig()
        self.silence_threshold = global_config.silence_threshold
        self.n_fft = global_config.n_fft
        self.hop_length = global_config.hop_length
        self.fmin = global_config.crepe_fmin
        self.fmax = global_config.crepe_fmax
        self.batch_size = global_config.crepe_batch_size
        self.filter_win_frames = global_config.crepe_filter_win_frames

    def expand(self):
        return self.silence_threshold, self.hop_length, self.fmin, self.fmax, self.batch_size, self.filter_win_frames


class NetworkConfig(Config):

    modes = {'cat', 'cat_attn', 'cross_attn', 'double_attn', 'cat_pre_post_attn'}

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mode = 'cat'
        self.embed_dim = 64
        self.attn_heads = 4
        self.dropout = 0
        self.recurrent_dropout = 0.3

    def validate_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(f'Invalid mode: {mode}')

        return True

    def update(self, **kwargs):
        if 'mode' in kwargs.keys():
            self.validate_mode(kwargs['mode'])
        super().update(**kwargs)


class PositionalEncodingConfig(Config):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AudioPlayerConfig(Config):

    def __init__(self):
        self.device = 'cuda'
        self.mode = 'play'
        self.method = 'griffin-lim'
        self.sr = 16000
        self.gan_sr = 22050
        self.n_fft = 1024
        self.n_mels = 80
        self.compression_factor = 1
        self.denoising_strength = 0.005
