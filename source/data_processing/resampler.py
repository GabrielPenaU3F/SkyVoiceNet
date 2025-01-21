import librosa
import numpy as np


class Resampler:

    def __init__(self, original_sr):
        self.original_sr = original_sr

    def resample(self, audio, new_sr):
        return librosa.resample(audio, orig_sr=self.original_sr, target_sr=new_sr)

    def resample_randomly(self, audio, min_factor, max_factor):
        resample_factor = np.random.uniform(min_factor, max_factor)
        new_sr = int(self.original_sr * resample_factor)
        resampled_speech = self.resample(audio, new_sr)
        return resampled_speech, new_sr
