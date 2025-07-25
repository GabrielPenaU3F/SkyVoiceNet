import numpy as np
from matplotlib import pyplot as plt

from source.data_management.data_loader import DataLoader
from source.data_processing.word_processor import WordProcessor

import sounddevice as sd

raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)
row = raw_data.iloc[0]

melody = row['sing']
speech = row['read']
melody_marks = row['marks_sing']
speech_marks = row['marks_read']

sing_phrases = WordProcessor.segment_phrases(melody_marks)  # Divide sing into phrases
sing_words_per_phrase = WordProcessor.segment_words(sing_phrases)  # Divide each frase into words
read_words_per_phrase = WordProcessor.match_read_segments(sing_words_per_phrase, speech_marks)

read_audios = WordProcessor.extract_audio_segments(speech, read_words_per_phrase)

phrase = read_audios[2]
audio = np.concatenate(phrase)

fft = np.abs(np.fft.fft(audio))[:int(len(audio)/2)]
f_axis = np.linspace(0, 22050, len(fft), endpoint=False)

plt.plot(f_axis, fft)
plt.show()