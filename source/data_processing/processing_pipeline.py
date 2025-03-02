import pandas as pd

from source.config import PreprocessConfig, CrepeConfig
from source.data_management.data_writer import DataWriter
from source.data_processing.contour_extractor import ContourExtractor
from source.data_processing.normalizer import Normalizer
from source.data_processing.resampler import Resampler
from source.data_processing.silence_remover import SilenceRemover
from source.data_processing.spectrogram_transformer import SpectrogramTransformer


class ProcessingPipeline:

    def __init__(self):
        self.config = PreprocessConfig() # This one is a singleton
        self.resampler = Resampler(self.config.original_sr)
        self.normalizer = Normalizer()
        self.silence_remover = SilenceRemover()
        self.spectrogram_transformer = SpectrogramTransformer()
        self.contour_extractor = ContourExtractor()

    def process(self, data, **kwargs):
        self.config.update(**kwargs)

        processed_data = self.preprocess(data)
        processed_data = self.postprocess(processed_data)

        if self.config.save:
            DataWriter().save_hdf5_processed(processed_data, self.config.filename)

        return processed_data

    def preprocess(self, data):

        preprocessed_data = pd.DataFrame(columns=['contour', 'melody_spectrogram', 'speech_spectrogram',
                                                  'speech_fs', 'melody_fs'])
        for index, row in data.iterrows():
            print(f'Preprocessing input element Nº {index + 1}')
            melody = row['sing']
            speech = row['read']

            # Resample
            resampled_melody = self.resampler.resample(melody, self.config.resample_sr)
            resampled_speech = self.resampler.resample(speech, self.config.resample_sr)

            # Normalize
            if self.config.normalize:
                resampled_melody = self.normalizer.normalize(resampled_melody)
                resampled_speech = self.normalizer.normalize(resampled_speech)

            # Remove silences and trails
            silenceless_speech = self.silence_remover.remove_silence(resampled_speech, self.config.resample_sr,
                                                                     self.config.silence_threshold,
                                                                     self.config.max_allowed_silence_duration)
            melody_no_trail = self.silence_remover.remove_trailing_silences(resampled_melody,
                                                                            self.config.silence_threshold,
                                                                            self.config.hop_length)

            # Resample randomly
            randomly_resampled_speech, speech_sr = self.resampler.resample_randomly(silenceless_speech,
                                                                                    self.config.min_resample_factor,
                                                                                    self.config.max_resample_factor)

            # Obtain spectrograms
            speech_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                randomly_resampled_speech, self.config.n_fft, self.config.hop_length, self.config.win_length)
            melody_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                melody_no_trail, self.config.n_fft, self.config.hop_length, self.config.win_length)

            melody_contour = self.contour_extractor.extract_contour(resampled_melody, self.config.resample_sr, CrepeConfig())

            # Time stretch
            speech_spectrogram = self.spectrogram_transformer.stretch_spectrogram(
                speech_spectrogram, melody_contour.shape[1])

            # Normalize
            if self.config.normalize:
                speech_spectrogram = self.normalizer.spectrogram_min_max_normalize(speech_spectrogram)
                melody_spectrogram = self.normalizer.spectrogram_min_max_normalize(melody_spectrogram)

            preprocessed_row = pd.DataFrame(
                {'contour': [melody_contour],
                 'melody_spectrogram': [melody_spectrogram],
                 'speech_spectrogram': [speech_spectrogram],
                 'melody_sr': [self.config.resample_sr],
                 'speech_sr': [speech_sr]},
                index=[len(preprocessed_data)])
            preprocessed_data = pd.concat([preprocessed_data, preprocessed_row], ignore_index=True)

        return preprocessed_data

    def postprocess(self, data):

        speech_spectrogram = data['speech_spectrogram'].values
        max_t = max([spectrogram.shape[1] for spectrogram in speech_spectrogram])
        for index, row in data.iterrows():
            print(f'Postprocessing input element Nº {index + 1}')

            contour_spectrogram = row['contour']
            speech_spectrogram = row['speech_spectrogram']
            melody_spectrogram = row['melody_spectrogram']
            contour_spectrogram = self.spectrogram_transformer.zeropad_time(contour_spectrogram, max_t, padding=self.config.padding)
            speech_spectrogram = self.spectrogram_transformer.zeropad_time(speech_spectrogram, max_t, padding=self.config.padding)
            melody_spectrogram = self.spectrogram_transformer.zeropad_time(melody_spectrogram, max_t, padding=self.config.padding)

            data.at[index, 'contour'] = contour_spectrogram
            data.at[index, 'speech_spectrogram'] = speech_spectrogram
            data.at[index, 'melody_spectrogram'] = melody_spectrogram

        return data
