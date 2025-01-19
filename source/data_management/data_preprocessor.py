import librosa
import torch
import torchcrepe
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from source import utilities
from source.data_management.data_writer import DataWriter
from source.singleton import Singleton


class DataPreprocessor(metaclass=Singleton):

    def __init__(self):
        self.config = PreprocessConfig()

    def preprocess(self, data, **kwargs):
        self.config.update(**kwargs)  # Actualizar parámetros con kwargs
        preprocessed_data = pd.DataFrame(columns=['contour', 'melody_spectrogram', 'speech_spectrogram',
                                                  'speech_fs', 'melody_fs'])
        for index, row in data.iterrows():

            print(f'Preprocessing input element Nº {index + 1}')
            melody = row['sing']
            speech = row['read']
            resampled_melody, resampled_speech = self.resample(melody, speech)

            norm_melody, norm_speech = self.normalize_values(resampled_melody, resampled_speech)

            silenceless_speech = self.remove_speech_silence(norm_speech)

            randomly_resampled_speech, speech_fs = self.random_resample(silenceless_speech)

            speech_spectrogram = self.obtain_log_spectrogram(randomly_resampled_speech)

            melody_contour = self.extract_melody_contour(norm_melody)

            # Espectrograma de la señal de audio original
            # melody = row['contour']
            # contour = melody_contour_row['contour'].values[0]
            # melody_spect = librosa.amplitude_to_db(np.abs(librosa.stft(melody, hop_length=256)), ref=np.max)
            # utilities.draw_spectrograms(melody_spect, contour)

            stretched_spectrogram = self.stretch_spectrogram(melody_contour, speech_spectrogram)

            melody_spectrogram = self.obtain_log_spectrogram(norm_melody)

            preprocessed_row = pd.DataFrame(
                {'contour': [melody_contour],
                 'melody_spectrogram': [melody_spectrogram],
                 'speech_spectrogram': [stretched_spectrogram],
                 'melody_fs': [self.config.resample_fs],
                 'speech_fs': [speech_fs]},
                index=[len(preprocessed_data)])
            preprocessed_data = pd.concat([preprocessed_data, preprocessed_row], ignore_index=True)

        if self.config.save:
            DataWriter().save_hdf5_preprocessed(preprocessed_data, self.config.filename)

        return preprocessed_data

    # Input: raw melody and speech
    # Output: resampled melody and speech
    def resample(self, melody, speech):
        melody = librosa.resample(melody, orig_sr=self.config.fs, target_sr=self.config.resample_fs)
        speech = librosa.resample(speech, orig_sr=self.config.fs, target_sr=self.config.resample_fs)
        return melody, speech

    # Input: resampled melody and speech
    # Output: normalized melody and speech
    def normalize_values(self, melody, speech):
        melody = librosa.util.normalize(melody)
        speech = librosa.util.normalize(speech)
        return melody, speech

    # Input: normalized speech
    # Output: speech without silence frames
    def remove_speech_silence(self, speech):

        # Divide audio into non-silent intervals
        non_silent_intervals = librosa.effects.split(speech, top_db=self.config.silence_threshold)
        # By default, this covers 50ms
        max_silence_samples = int(self.config.max_allowed_silence_duration * self.config.resample_fs)
        silent_intervals = self.obtain_silent_intervals(speech, non_silent_intervals)
        # This filters the array to identify long silences
        filtered_silent_intervals = [
            (start, end) for start, end in silent_intervals if (end - start) >= max_silence_samples
        ]

        # Finally, we rebuild the audio omitting the long silences
        filtered_fragments = self.rebuild_audio_without_long_silences(speech, filtered_silent_intervals)
        clean_speech = np.concatenate(filtered_fragments)

        return clean_speech

    def rebuild_audio_without_long_silences(self, speech, filtered_silent_intervals):
        filtered_audio = []
        prev_end = 0
        for start, end in filtered_silent_intervals:
            filtered_audio.append(speech[prev_end:start])
            prev_end = end
        if prev_end < len(speech):
            filtered_audio.append(speech[prev_end:])
        return filtered_audio

    def obtain_silent_intervals(self, speech, non_silent_intervals):
        silent_intervals = []
        start_index = 0
        for interval in non_silent_intervals:
            # Add the next silent interval
            if start_index < interval[0]:
                silent_intervals.append([start_index, interval[0] - 1])

            start_index = interval[1] + 1
        # Add the last silent interval, if it exists
        if start_index < len(speech):
            silent_intervals.append([start_index, len(speech) - 1])

        return silent_intervals

    # Input: speech
    # Output: randomly resampled and sample frequency
    def random_resample(self, speech):
        resample_factor = np.random.uniform(self.config.min_resample_factor, self.config.max_resample_factor)
        new_fs = int(self.config.resample_fs * resample_factor)
        resampled_speech = librosa.resample(speech, orig_sr=self.config.resample_fs, target_sr=new_fs)
        return resampled_speech, new_fs

    # Input: randomly resampled speech or normalized melody
    # Output: speech or melody spectrogram
    def obtain_log_spectrogram(self, audio):
        spectrogram = librosa.stft(audio, n_fft=self.config.n_fft,
                                          hop_length=self.config.hop_length, win_length=self.config.win_length)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
        return spectrogram_db

    # Input: melody and speech spectrogram
    # Output: contour spectrogram and speech spectrogram
    def extract_melody_contour(self, melody):
        # Make sure the audio is mono
        if melody.ndim > 1:
            melody = librosa.to_mono(melody)
        # Remove trailing silences and normalize to [-1, 1]
        melody_trim, melody_idx = librosa.effects.trim(melody, top_db=self.config.silence_threshold,
                                                       hop_length=self.config.hop_length)
        melody = melody_trim / np.max(np.abs(melody_trim))
        # Convert to torch tensor with the necessary dimensions
        melody = torch.from_numpy(melody)[None, :].to(device='cuda')
        torch.cuda.empty_cache()
        contour = torchcrepe.predict(melody, self.config.resample_fs, hop_length=self.config.hop_length, model='full',
                                     device='cuda:0', fmin=self.config.crepe_fmin, fmax=self.config.crepe_fmax,
                                     batch_size=self.config.crepe_batch_size)
         # Filter with a window of 3 frames (~48 ms)
        smooth_contour = torchcrepe.filter.median(contour, win_length=self.config.crepe_filter_win_frames)
        midi_contour = utilities.convert_to_midi_note(smooth_contour)
        contour = self.build_midi_contour_spectrogram(midi_contour)
        return contour

    def build_midi_contour_spectrogram(self, midi_contour):

        # This will manually 'build' a spectrogram using the time and frequency values
        # obtained as CREPE outputs
        midi_contour = midi_contour.cpu().numpy().squeeze()
        num_frames = len(midi_contour)
        contour_spectrogram = np.zeros((128, num_frames), dtype=np.float32)
        frame_indices = np.arange(num_frames)

        # Fill the spectrogram
        for t_idx, f_idx in zip(frame_indices, midi_contour):
            contour_spectrogram[f_idx, t_idx] += 1

        return contour_spectrogram

    # Input: melody contour and speech spectrogram
    # Output: speech spectrogram with the same length as the melody contour
    def stretch_spectrogram(self, contour, speech_spectrogram):
        contour_length = contour.shape[1]
        speech_length = speech_spectrogram.shape[1]
        scale_factor = contour_length / speech_length
        stretched_speech_spectrogram = zoom(speech_spectrogram, (1, scale_factor), order=1)
        # utilities.draw_spectrograms(stretched_speech_spectrogram, contour)
        return stretched_speech_spectrogram



class PreprocessConfig(metaclass=Singleton):

    def __init__(self):
        self.fs = 44100
        self.resample_fs = 16000
        self.silence_threshold = 40
        self.max_allowed_silence_duration = 0.05
        self.target_fs = 16000
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

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid parameter: {key}')
