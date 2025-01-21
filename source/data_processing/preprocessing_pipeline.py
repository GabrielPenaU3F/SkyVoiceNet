import pandas as pd

from source.config import PreprocessConfig, CrepeConfig
from source.data_management.data_writer import DataWriter
from source.data_processing.contour_extractor import ContourExtractor
from source.data_processing.normalizer import Normalizer
from source.data_processing.resampler import Resampler
from source.data_processing.silence_remover import SilenceRemover
from source.data_processing.spectrogram_transformer import SpectrogramTransformer


class PreprocessingPipeline:

    def __init__(self):
        self.config = PreprocessConfig() # This one is a singleton
        self.resampler = Resampler(self.config.original_sr)
        self.normalizer = Normalizer()
        self.silence_remover = SilenceRemover()
        self.spectrogram_transformer = SpectrogramTransformer()
        self.contour_extractor = ContourExtractor()

    def preprocess(self, data, **kwargs):
        self.config.update(**kwargs)  # Actualize parameters
        preprocessed_data = pd.DataFrame(columns=['contour', 'melody_spectrogram', 'speech_spectrogram',
                                                  'speech_fs', 'melody_fs'])
        for index, row in data.iterrows():

            print(f'Preprocessing input element Nº {index + 1}')
            melody = row['sing']
            speech = row['read']

            resampled_melody = self.resampler.resample(melody, self.config.resample_sr)
            resampled_speech = self.resampler.resample(speech, self.config.resample_sr)

            norm_melody = self.normalizer.normalize(resampled_melody)
            norm_speech = self.normalizer.normalize(resampled_speech)

            silenceless_speech = self.silence_remover.remove_silence(norm_speech, self.config.resample_sr,
                                                                     self.config.silence_threshold,
                                                                     self.config.max_allowed_silence_duration)
            melody_no_trail = self.silence_remover.remove_trailing_silences(norm_melody,
                                                                            self.config.silence_threshold,
                                                                            self.config.hop_length)

            randomly_resampled_speech, speech_sr = self.resampler.resample_randomly(silenceless_speech,
                                                                                    self.config.min_resample_factor,
                                                                                    self.config.max_resample_factor)

            speech_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                randomly_resampled_speech, self.config.n_fft, self.config.hop_length, self.config.win_length)
            melody_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                melody_no_trail, self.config.n_fft, self.config.hop_length, self.config.win_length)

            melody_contour = self.contour_extractor.extract_contour(norm_melody, self.config.resample_sr, CrepeConfig())

            stretched_spectrogram = self.spectrogram_transformer.stretch_spectrogram(
                speech_spectrogram, melody_contour.shape[1])

            preprocessed_row = pd.DataFrame(
                {'contour': [melody_contour],
                 'melody_spectrogram': [melody_spectrogram],
                 'speech_spectrogram': [stretched_spectrogram],
                 'melody_sr': [self.config.resample_sr],
                 'speech_sr': [speech_sr]},
                index=[len(preprocessed_data)])
            preprocessed_data = pd.concat([preprocessed_data, preprocessed_row], ignore_index=True)

        if self.config.save:
            DataWriter().save_hdf5_preprocessed(preprocessed_data, self.config.filename)

        return preprocessed_data
