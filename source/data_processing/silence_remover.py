import librosa
import numpy as np


class SilenceRemover:


    def remove_silence(self, audio, sr, threshold, max_allowed_silence_duration):

        # Divide audio into non-silent intervals
        non_silent_intervals = librosa.effects.split(audio, top_db=threshold)
        max_silence_samples = int(max_allowed_silence_duration * sr)
        silent_intervals = self.obtain_silent_intervals(audio, non_silent_intervals)
        # This filters the array to identify long silences
        filtered_silent_intervals = [
            (start, end) for start, end in silent_intervals if (end - start) >= max_silence_samples
        ]

        # Finally, we rebuild the audio omitting the long silences
        filtered_fragments = self.rebuild_audio_without_long_silences(audio, filtered_silent_intervals)
        clean_speech = np.concatenate(filtered_fragments)

        return clean_speech

    def rebuild_audio_without_long_silences(self, audio, filtered_silent_intervals):
        filtered_audio = []
        prev_end = 0
        for start, end in filtered_silent_intervals:
            filtered_audio.append(audio[prev_end:start])
            prev_end = end
        if prev_end < len(audio):
            filtered_audio.append(audio[prev_end:])
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

    def remove_trailing_silences(self, audio, threshold, hop_length):
        audio_trim, _ = librosa.effects.trim(audio, top_db=threshold, hop_length=hop_length)
        return audio_trim
