import os

import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

from source.data_management.data_loader import DataLoader
from source.data_management.path_repo import PathRepo
from source.data_processing.processing_pipeline import ProcessingPipeline
from source.data_processing.word_processor import WordProcessor

proc = ProcessingPipeline()

raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)

# for index, row in raw_data.iterrows():

row = raw_data.iloc[0]

melody = row['sing']
speech = row['read']
melody_marks = row['marks_sing']
speech_marks = row['marks_read']

# Segment
sing_phrases = WordProcessor.segment_phrases(melody_marks)  # Divide sing into phrases
sing_words_per_phrase = WordProcessor.segment_words(sing_phrases)  # Divide each frase into words
read_words_per_phrase = WordProcessor.match_read_segments(sing_words_per_phrase, speech_marks)

sing_audios = WordProcessor.extract_audio_segments(melody, sing_words_per_phrase)
speech_audios = WordProcessor.extract_and_stretch_speech_segments(
    speech_audio=speech,
    speech_segments=read_words_per_phrase,
    sing_segments=sing_words_per_phrase,
    sr=44100
)

for idx, audio_phrase in enumerate(speech_audios):

    # if idx == 2:
    #     import sounddevice as sd
    #     sd.play(audio_phrase[0])
    #     sd.wait()
    #
    #     plt.plot(audio_phrase[0])
    #     plt.show()

    audio_phrase = np.concatenate(audio_phrase)
    audio_phrase = np.concatenate((audio_phrase, np.zeros(int(44100/10))))

    path = PathRepo().get_test_wads_path()
    scaled = (audio_phrase * 32767).astype(np.int16)
    wavfile.write(os.path.join(path, str(idx) + '.wav'), 44100, scaled)
