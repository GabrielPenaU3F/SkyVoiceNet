import numpy as np

from source.data_management.data_loader import DataLoader
from source.data_processing.processing_pipeline import ProcessingPipeline
from source.data_processing.word_processor import WordProcessor

import sounddevice as sd


def extract_phrases_and_words(audio, segments, sr=44100):
    """
    Extrae:
    - Cada frase completa como un clip de audio
    - Cada palabra individual como clips separados

    Retorna:
    - phrase_audio: [audio_clip_frase_1, audio_clip_frase_2, ...]
    - word_audio: [[palabra_1_clip, palabra_2_clip, ...], [...], ...]
    """
    phrase_audio = []
    word_audio = []

    for phrase_idx, phrase in enumerate(segments):
        # Frase: lista de palabras (cada palabra es lista de fonemas)
        if not phrase:
            continue  # evitar errores si la frase está vacía

        try:
            # Frase completa
            phrase_start = phrase[0][0]["start"]
            phrase_end = phrase[-1][-1]["end"]

            start_sample = int(phrase_start * sr)
            end_sample = int(phrase_end * sr)

            if start_sample < 0 or end_sample > len(audio) or start_sample >= end_sample:
                print(f"[FRASE {phrase_idx + 1}] Segmento inválido: start={phrase_start}, end={phrase_end}")
                continue

            phrase_clip = audio[start_sample:end_sample]
            phrase_audio.append(phrase_clip)

            # Palabras individuales
            word_clips = []
            for word_idx, word in enumerate(phrase):
                if not word:
                    continue

                word_start = word[0]["start"]
                word_end = word[-1]["end"]

                ws = int(word_start * sr)
                we = int(word_end * sr)

                if ws < 0 or we > len(audio) or ws >= we:
                    print(
                        f"[FRASE {phrase_idx + 1} - PALABRA {word_idx + 1}] Segmento inválido: start={word_start}, end={word_end}")
                    continue

                word_clips.append(audio[ws:we])

            word_audio.append(word_clips)

        except Exception as e:
            print(f"Error al procesar frase {phrase_idx + 1}: {e}")
            continue

    return phrase_audio, word_audio

def match_read_segments_by_count(sing_words_per_phrase, read_marks):
    """
    Alinea las palabras del canto con el habla, por conteo de fonemas.
    Ignora un 'sil' inicial y asigna los 'sil' entre palabras como cierre de la palabra anterior.
    """
    read_segments = []
    read_marks_list = list(read_marks.to_dict(orient="records"))
    read_index = 0

    # Ignorar sil inicial
    if read_marks_list and read_marks_list[0]["mark"] == "sil":
        read_index += 1

    for phrase_idx, phrase in enumerate(sing_words_per_phrase):
        read_phrase = []

        for word_idx, word in enumerate(phrase):
            num_phonemes = len(word)
            word_segment = []

            while len(word_segment) < num_phonemes and read_index < len(read_marks_list):
                word_segment.append(read_marks_list[read_index])
                read_index += 1

            # Incluir un 'sil' siguiente si existe
            if read_index < len(read_marks_list):
                next_entry = read_marks_list[read_index]
                if next_entry["mark"] == "sil":
                    word_segment.append(next_entry)
                    read_index += 1

            read_phrase.append(word_segment)

        read_segments.append(read_phrase)

    return read_segments


proc = ProcessingPipeline()

raw_data = DataLoader.load_raw_data('nus_data_raw.h5', load_wavs=False)

for index, row in raw_data.iterrows():

    melody = row['sing']
    speech = row['read']
    melody_marks = row['marks_sing']
    speech_marks = row['marks_read']

    # Segment
    sing_phrases = WordProcessor.segment_phrases(melody_marks)  # Divide sing into phrases
    sing_words_per_phrase = WordProcessor.segment_words(sing_phrases)  # Divide each frase into words
    read_words_per_phrase = match_read_segments_by_count(sing_words_per_phrase, speech_marks)

    sing_phrase_audios, sing_word_audios = extract_phrases_and_words(melody, sing_words_per_phrase)
    read_phrase_audios , read_word_audios = extract_phrases_and_words(speech, read_words_per_phrase)

    words = [x for i in read_words_per_phrase for x in i]
    for idx, word in enumerate(read_word_audios):
        a = 2
        duration = words[idx][-1]['end'] - words[idx][0]['start']
        print(f'Word duration: {duration:.4f} seconds')
        sd.play(word[0], samplerate=44100)
        sd.wait()
