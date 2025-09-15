import numpy as np
import torch

from pesq import pesq
from pystoi import stoi
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

from source.data_processing.contour_extractor import ContourExtractor


class MetricCalculator:

    def __init__(self):
        pass

    def psnr(self, spectrogram_ref, spectrogram_test):
        mse = np.mean((spectrogram_ref - spectrogram_test) ** 2)
        if mse == 0:
            return 0
        max_val = np.max(spectrogram_ref)
        return 20 * np.log10(max_val / np.sqrt(mse))

    def pesq(self, audio_ref, audio_test, sr=16000):
        return pesq(sr, audio_ref, audio_test, 'wb')  # 'wb' = wideband

    def si_sdr(self, audio_ref, audio_test):
        return scale_invariant_signal_distortion_ratio(torch.tensor(audio_ref), torch.tensor(audio_test))

    def f0_rmse(self, audio_ref, audio_test, sr=16000):
        extractor = ContourExtractor()
        contour_ref = extractor.extract_contour(audio_ref, sr, mode=None, trim=False).squeeze().cpu().numpy()
        contour_test = extractor.extract_contour(audio_test, sr, mode=None, trim=False).squeeze().cpu().numpy()
        return np.sqrt(((contour_ref - contour_test) ** 2).mean())

    def stoi(self, audio_ref, audio_test, sr=16000):
        return stoi(audio_ref, audio_test, sr, extended=False)