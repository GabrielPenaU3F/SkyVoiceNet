import librosa
import numpy as np
from librosa import stft, istft
from librosa.effects import time_stretch
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import firwin, lfilter


class WordProcessor:


    # Divide text in phrases by using 'sil' as separator
    # Sil indicates large silences between phrases in the sing marks
    @staticmethod
    def segment_phrases(marks):

        phrases = []
        current_phrase = []

        for _, row in marks.iterrows():
            if row["mark"] == "sil":
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []
            else:
                current_phrase.append(row.to_dict())

        if current_phrase:
            phrases.append(current_phrase)

        return phrases

    # Divide phrases into words by using "sp" as separator
    # Sp indicates short pause between words in sing marks
    @staticmethod
    def segment_words(phrases):

        words_per_phrase = []

        for phrase in phrases:
            words = []
            current_word = []

            for entry in phrase:
                if entry["mark"] == "sp":
                    if current_word:
                        words.append(current_word)
                        current_word = []
                else:
                    current_word.append(entry)

            if current_word:
                words.append(current_word)

            words_per_phrase.append(words)

        return words_per_phrase

    @staticmethod
    def match_read_segments(sing_words_per_phrase, read_marks):

        # Eliminar fonemas 'sil' de read_marks antes de empezar la iteración
        read_marks = read_marks[read_marks["mark"] != "sil"].reset_index(drop=True)

        read_segments = []
        read_index = 0

        for phrase in sing_words_per_phrase:
            read_phrase = []
            read_word = []
            for word in phrase:
                for phoneme in word:
                    if read_index < len(read_marks):

                        if phoneme["mark"] == "sp":
                            if read_word:
                                read_phrase.append(read_word)
                            read_word = []
                        else:
                            read_word.append(read_marks.iloc[read_index].to_dict())
                        read_index += 1

                if read_word:
                    read_phrase.append(read_word)
                    read_word = []

            read_segments.append(read_phrase)

        return read_segments

    # Generate all possible combinations of three or more words
    @staticmethod
    def generate_combinations(words_per_phrase):
        segments = []

        for words in words_per_phrase:
            if len(words) >= 3:
                for i in range(len(words)):
                    for j in range(i + 3, len(words) + 1):
                        segments.append([word for word in words[i:j]])

        return segments

    @staticmethod
    def extract_audio_segments(audio, segments, sr=44100):
        extracted = []

        for phrase_idx, phrase in enumerate(segments):
            phrase_audio = []

            for word_idx, word in enumerate(phrase):
                if not word:
                    continue

                start = word[0]["start"]
                end = word[-1]["end"]

                start_idx = int(round(start * sr))
                end_idx = int(round(end * sr))

                if start_idx < 0 or end_idx > len(audio) or start_idx >= end_idx:
                    print(
                        f"[WARNING] Segmento inválido: frase {phrase_idx}, palabra {word_idx}, start={start}, end={end}")
                    continue

                audio_segment = audio[start_idx:end_idx]
                phrase_audio.append(audio_segment)

            extracted.append(phrase_audio)

        return extracted

    @staticmethod
    def extract_and_stretch_speech_segments(speech_audio, speech_segments, sing_segments, sr=44100):
        """Estira cada fonema hablado para empatar su duración con el correspondiente del canto."""

        stretched_words_per_phrase = []

        for phrase_idx, (read_phrase, sing_phrase) in enumerate(zip(speech_segments, sing_segments)):
            stretched_phrase = []

            for word_idx, (read_word, sing_word) in enumerate(zip(read_phrase, sing_phrase)):
                stretched_phonemes = []

                for read_phoneme, sing_phoneme in zip(read_word, sing_word):
                    t_read_start = read_phoneme['start']
                    t_read_end = read_phoneme['end']
                    t_sing_start = sing_phoneme['start']
                    t_sing_end = sing_phoneme['end']

                    dur_read = t_read_end - t_read_start
                    dur_sing = t_sing_end - t_sing_start

                    # Evitamos problemas numéricos
                    if dur_read <= 0 or dur_sing <= 0:
                        print(f"[WARNING] Fonema con duración inválida en palabra {word_idx} de frase {phrase_idx}. Se omite.")
                        continue

                    stretch_ratio = dur_read / (1e-3 + dur_sing)

                    start_idx = int(round(t_read_start * sr))
                    end_idx = int(round(t_read_end * sr))
                    phoneme_audio = speech_audio[start_idx:end_idx]

                    # Estiramos
                    try:
                        stretched = WordProcessor.safe_time_stretch(phoneme_audio.astype(float), rate=stretch_ratio)
                        adjusted = WordProcessor.match_energy(phoneme_audio, stretched)
                        stretched_phonemes.append(adjusted)
                    except Exception as e:
                        print(f"[ERROR] Fallo al estirar fonema en {word_idx} de frase {phrase_idx}: {e}")
                        continue

                # Concatenar los fonemas estirados para formar la palabra
                if stretched_phonemes:
                    smooth_phonemes = WordProcessor.smooth_transition(stretched_phonemes, transition_size=200)
                    word_audio = WordProcessor.crossfade_concat(smooth_phonemes, fade_samples=300)
                    stretched_phrase.append(word_audio)

            stretched_words_per_phrase.append(stretched_phrase)

        return stretched_words_per_phrase

    @staticmethod
    def safe_time_stretch(y, rate, n_fft=2048):

        # n_fft no debe exceder el tamaño de la señal
        n_fft = min(n_fft, len(y) - (len(y) % 2))  # par más cercano

        hop_length = n_fft // 4  # estándar en librosa
        D = stft(y, n_fft=n_fft, hop_length=hop_length)
        D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
        y_stretched = istft(D_stretched, hop_length=hop_length, length=int(len(y) / rate))
        return y_stretched

    @staticmethod
    def smooth_transition(word, transition_size=200):
        smooth_word = word.copy()

        for i in range(len(word) - 1):
            f1 = smooth_word[i]
            f2 = smooth_word[i + 1]

            # Evitar errores si el fonema es más corto que la transición
            n = min(transition_size, len(f1), len(f2))

            # Extraer segmentos a suavizar
            tail = f1[-n:]
            head = f2[:n]

            # Promedio con suavizado lineal
            weights = np.linspace(0, 1, n)
            blended = (1 - weights) * tail + weights * head

            # Reemplazar en los fonemas originales
            smooth_word[i] = np.concatenate([f1[:-n], blended])
            smooth_word[i + 1] = np.concatenate([blended, f2[n:]])

        return smooth_word

    @staticmethod
    def crossfade_concat(segments, fade_samples=100):
        """
        Concatena fonemas aplicando crossfade entre ellos.
        """
        result = segments[0]
        for i in range(1, len(segments)):
            prev = result
            curr = segments[i]

            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)

            overlap = prev[-fade_samples:] * fade_out + curr[:fade_samples] * fade_in

            # Armar nueva señal
            combined = np.concatenate([
                prev[:-fade_samples],
                overlap,
                curr[fade_samples:]
            ])
            result = combined
        return result

    @staticmethod
    def match_energy(original, modified):
        """
        Ajusta la energía (RMS) del segmento modificado para que coincida con el original.
        """
        rms_orig = np.sqrt(np.mean(original**2))
        rms_mod = np.sqrt(np.mean(modified**2))
        if rms_mod == 0:
            return modified  # evitar división por cero
        gain = rms_orig / rms_mod
        return modified * gain