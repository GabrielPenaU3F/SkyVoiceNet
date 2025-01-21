import librosa


class Normalizer:

    def normalize(self, audio):
        return librosa.util.normalize(audio)
