import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
from matplotlib import pyplot as plt

from source.config import PreprocessConfig
from source.data_management.data_writer import DataWriter
from source.data_processing.contour_extractor import ContourExtractor
from source.data_processing.normalizer import Normalizer
from source.data_processing.resampler import Resampler
from source.data_processing.spectrogram_transformer import SpectrogramTransformer
from source.data_processing.word_processor import WordProcessor


class ProcessingPipeline:

    def __init__(self):
        self.config = PreprocessConfig() # This one is a singleton
        self.resampler = Resampler(self.config.original_sr)
        self.normalizer = Normalizer()
        self.spectrogram_transformer = SpectrogramTransformer()
        self.contour_extractor = ContourExtractor()

    def process(self, data, **kwargs):
        self.config.update(**kwargs)
        processed_data = None

        if self.config.mode == 'pre':
            processed_data = self.preprocess(data)
        elif self.config.mode == 'post':
            processed_data = self.postprocess(data)
        elif self.config.mode == 'full':
            processed_data = self.preprocess(data)
            processed_data = self.postprocess(processed_data)

        if self.config.save:
            DataWriter().save_hdf5_processed(processed_data, self.config.filename)

        return processed_data

    def preprocess(self, data):

        preprocessed_data = pd.DataFrame(columns=['contour', 'song', 'speech'])
        for index, row in data.iterrows():
            print(f'Preprocessing audio element Nº {index + 1}')
            melody = row['sing']
            speech = row['read']
            melody_marks = row['marks_sing']
            speech_marks = row['marks_read']

            # Segment
            sing_phrases = WordProcessor.segment_phrases(melody_marks)  # Divide sing into phrases
            sing_words_per_phrase = WordProcessor.segment_words(sing_phrases)  # Divide each frase into words
            read_words_per_phrase = WordProcessor.match_read_segments(sing_words_per_phrase,
                                                                      speech_marks)  # Divide read into words

            # Extract segments from audios
            sing_audio_words_per_phrase = WordProcessor.extract_audio_segments(melody, sing_words_per_phrase)
            speech_audio_words_per_phrase = WordProcessor.extract_and_stretch_speech_segments(
                speech_audio=speech,
                speech_segments=read_words_per_phrase,
                sing_segments=sing_words_per_phrase,
                sr=44100
            )

            # Generate combinations
            sing_audio_segments = WordProcessor.generate_combinations(sing_audio_words_per_phrase)
            read_audio_segments = WordProcessor.generate_combinations(speech_audio_words_per_phrase)

            # for idx, audio_segment in enumerate(read_audio_segments):
            #
            #     if(len(audio_segment) == 5):
            #         audio = np.concatenate(audio_segment)
            #         fft = np.abs(np.fft.fft(audio))[:int(len(audio) / 2)]
            #         f_axis = np.linspace(0, 22050, len(fft), endpoint=False)
            #
            #         plt.plot(f_axis, fft)
            #         plt.show()
            #         sd.play(audio, 44100)
            #         sd.wait()

            # Resample, extract contour and add
            for idx, (sing_audio, read_audio) in enumerate(zip(sing_audio_segments, read_audio_segments)):
                sing_audio = np.concatenate(sing_audio)
                read_audio = np.concatenate(read_audio)
                resampled_song = self.resampler.resample(sing_audio, self.config.resample_sr)
                resampled_speech = self.resampler.resample(read_audio, self.config.resample_sr)
                melody_contour = self.contour_extractor.extract_contour(resampled_song, sr=self.config.resample_sr)

                preprocessed_row = pd.DataFrame(
                    {'contour': [melody_contour],
                     'song': [resampled_song],
                     'speech': [resampled_speech]
                     }
                )
                preprocessed_data = pd.concat([preprocessed_data, preprocessed_row], ignore_index=True)

            # Resample randomly
            # randomly_resampled_speech, speech_sr = self.resampler.resample_randomly(silenceless_speech,
            #                                                                         self.config.min_resample_factor,
            #                                                                         self.config.max_resample_factor)

        return preprocessed_data

    def postprocess(self, data):

        postprocessed_data = pd.DataFrame(columns=['contour', 'song', 'speech'])
        for index, row in data.iterrows():
            print(f'Postprocessing input element Nº {index + 1}')
            speech = row['speech']
            song = row['song']
            contour = row['contour']

            speech = self.match_melody_length(speech, song)

            # Obtain spectrograms
            speech_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                speech, self.config.n_fft, self.config.hop_length, self.config.win_length, self.config.keep_last_freq)
            melody_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                song, self.config.n_fft, self.config.hop_length, self.config.win_length, self.config.keep_last_freq)

            # Add
            postprocessed_row = pd.DataFrame(
                {'contour': [contour],
                 'song': [melody_spectrogram],
                 'speech': [speech_spectrogram]
                 }
            )
            postprocessed_data = pd.concat([postprocessed_data, postprocessed_row], ignore_index=True)

        return postprocessed_data

    def match_melody_length(self, speech, song):
        n = len(speech)
        m = len(song)
        if n < m:
            rate = n/m
            speech = librosa.effects.time_stretch(speech, rate=rate)
            return speech
        else: return speech
