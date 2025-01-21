import librosa
import numpy as np
import torch
import torchcrepe

from source import utilities


class ContourExtractor:

    def extract_contour(self, audio, sr, crepe_config):

        silence_threshold, hop_length, fmin, fmax, batch_size, filter_win_frames = crepe_config.expand()

        # Make sure the audio is mono
        if audio.ndim > 1:
            melody = librosa.to_mono(audio)

        # Remove trailing silences and normalize to [-1, 1]
        audio_trim, _ = librosa.effects.trim(audio, top_db=silence_threshold, hop_length=hop_length)
        audio_trim = audio_trim / np.max(np.abs(audio_trim))
        # Convert to torch tensor with the necessary dimensions
        audio_torch = torch.from_numpy(audio_trim)[None, :].to(device='cuda')
        torch.cuda.empty_cache()
        contour = torchcrepe.predict(audio_torch, sr, hop_length=hop_length, model='full', device='cuda:0',
                                     fmin=fmin, fmax=fmax, batch_size=batch_size)
        # Filter with a window of 3 frames (~48 ms)
        smooth_contour = torchcrepe.filter.median(contour, win_length=filter_win_frames)
        midi_contour = utilities.convert_to_midi_note(smooth_contour)
        contour = self.build_contour_spectrogram(midi_contour)
        return contour

    def build_contour_spectrogram(self, contour):

        # This will manually 'build' a spectrogram using the time and frequency values
        # obtained as CREPE outputs
        midi_contour = contour.cpu().numpy().squeeze()
        num_frames = len(midi_contour)
        contour_spectrogram = np.zeros((128, num_frames), dtype=np.float32)
        frame_indices = np.arange(num_frames)

        # Fill the spectrogram
        for t_idx, f_idx in zip(frame_indices, midi_contour):
            contour_spectrogram[f_idx, t_idx] += 1

        return contour_spectrogram
