import pandas as pd

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
            sing_phrases = WordProcessor.segment_phrases(melody_marks) # Divide sing into phrases
            sing_words_per_phrase = WordProcessor.segment_words(sing_phrases) # Divide each frase into words
            read_words_per_phrase = WordProcessor.match_read_segments(sing_words_per_phrase, speech_marks) # Divide read into words

            # Generate combinations
            sing_segments = WordProcessor.generate_combinations(sing_words_per_phrase)
            speech_segments = WordProcessor.generate_combinations(read_words_per_phrase)

            # Extract segments from audios
            sing_audio_segments = WordProcessor.extract_audio_segments(melody, sing_segments)
            read_audio_segments = WordProcessor.extract_audio_segments(speech, speech_segments)

            # for idx, audio_segment in enumerate(read_audio_segments):
            #     print(speech_segments[idx])
            #     sd.play(audio_segment, 44100)
            #     sd.wait()


            # Resample, extract contour and add
            for idx, (sing_audio, read_audio) in enumerate(zip(sing_audio_segments, read_audio_segments)):
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

            # Obtain spectrograms
            speech_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                speech, self.config.n_fft, self.config.hop_length, self.config.win_length)
            melody_spectrogram = self.spectrogram_transformer.obtain_log_spectrogram(
                song, self.config.n_fft, self.config.hop_length, self.config.win_length)

            # Time stretch speech
            speech_spectrogram = self.spectrogram_transformer.stretch_spectrogram(
                speech_spectrogram, contour.shape[1])

            # Normalize
            if self.config.normalize:
                speech_spectrogram = self.normalizer.spectrogram_min_max_normalize(speech_spectrogram)
                melody_spectrogram = self.normalizer.spectrogram_min_max_normalize(melody_spectrogram)

            # Add
            postprocessed_row = pd.DataFrame(
                {'contour': [contour],
                 'song': [melody_spectrogram],
                 'speech': [speech_spectrogram]
                 }
            )
            postprocessed_data = pd.concat([postprocessed_data, postprocessed_row], ignore_index=True)

        return postprocessed_data
