class WordProcessor:


    # Divide text in phrases by using 'sil' as separator
    # Sil indicates large silences between phrases in the sing marks
    @staticmethod
    def segment_phrases(marks):

        phrases = []
        current_phrase = []

        for _, row in marks.iterrows():

            mark = row["mark"]
            if not current_phrase and mark == "sil":
                continue  # ignore first 'sil'

            current_phrase.append(row.to_dict())
            if row["mark"] == "sil":
                if current_phrase:
                    phrases.append(current_phrase)
                    current_phrase = []

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
                if entry["mark"] == "sp" or entry['mark'] == 'sil':
                    current_word.append(entry) # Include the "sp" and "sil" phonemes
                    if current_word:
                        words.append(current_word)
                        current_word = []
                else:
                    current_word.append(entry)

            if current_word:
                words.append(current_word)

            words_per_phrase.append(words)

        return words_per_phrase

    # @staticmethod
    # def match_read_segments(sing_words_per_phrase, read_marks):
    #
    #     # Eliminar fonemas 'sil' de read_marks antes de empezar la iteración
    #     read_marks = read_marks[read_marks["mark"] != "sil"].reset_index(drop=True)
    #
    #     read_segments = []
    #     read_index = 0
    #
    #     for phrase in sing_words_per_phrase:
    #         read_phrase = []
    #         read_word = []
    #         for word in phrase:
    #             for phoneme in word:
    #                 if read_index < len(read_marks):
    #
    #                     if phoneme["mark"] == "sp":
    #                         if read_word:
    #                             read_phrase.append(read_word)
    #                         read_word = []
    #                     else:
    #                         read_word.append(read_marks.iloc[read_index].to_dict())
    #                     read_index += 1
    #
    #             if read_word:
    #                 read_phrase.append(read_word)
    #                 read_word = []
    #
    #         read_segments.append(read_phrase)
    #
    #     return read_segments

    @staticmethod
    def match_read_segments(sing_words_per_phrase, read_marks):
        """
        Alinea las palabras del canto con el habla, por conteo de fonemas.
        Ignora un 'sil' inicial y asigna los 'sil' entre palabras como cierre de la palabra anterior.
        """
        read_segments = []
        read_marks_list = list(read_marks.to_dict(orient="records"))
        read_index = 0

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

        for segment in segments:
            start = segment[0][0]["start"]
            end = segment[-1][-1]["end"]

            # Validar los valores de inicio y fin
            if start < 0 or end > len(audio) / sr or start >= end:
                print(f"Segmento inválido: start={start}, end={end}")
                continue

            extracted.append(audio[int(start * sr):int(end * sr)])

        return extracted
