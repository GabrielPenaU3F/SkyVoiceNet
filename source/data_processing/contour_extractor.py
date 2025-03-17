import librosa
import numpy as np
import torch
import torchcrepe

from source.config import CrepeConfig


class ContourExtractor:

    def __init__(self):
        self.config = CrepeConfig()

    def extract_contour(self, audio, sr=16000, num_bins=512):

        silence_threshold, hop_length, fmin, fmax, batch_size, filter_win_frames = self.config.expand()

        # Make sure the audio is mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

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
        contour = self.build_contour_spectrogram(smooth_contour, num_bins=num_bins)
        return contour


    def build_contour_spectrogram(self, contour, num_bins=512, sr=16000):
        """
        Construye una imagen binaria de tamaño (num_bins x T) con el contorno de la melodía de CREPE.

        Parámetros:
          - contour_hz: (T_c,) Array con las frecuencias de la melodía en Hz (salida de CREPE).
          - spec_shape: (num_bins, T_s) Dimensiones del espectrograma al que queremos ajustarnos.
          - sr: Frecuencia de muestreo del audio (default: 16000 Hz).
          - num_bins: Número de bins de frecuencia del espectrograma.

        Retorna:
          - contour_spectrogram: (num_bins, T_s) Imagen binaria con la melodía marcada.
        """
        num_frames = contour.shape[-1]

        # Convertimos el contorno a escala MIDI
        midi_contour = 69 + 12 * torch.log2(contour / 440)
        midi_contour = torch.clamp(torch.round(midi_contour), 0, 127).to(torch.float32)
        midi_contour = midi_contour.cpu().numpy().squeeze()

        # 3. Creamos la imagen binaria
        contour_spectrogram = np.zeros((num_bins, num_frames), dtype=np.float32)

        # Llenamos la imagen binaria
        for t_idx, midi_note in enumerate(midi_contour):
            bin_idx = self.midi_to_stft_bin(midi_note, sr, self.config.n_fft)
            contour_spectrogram[bin_idx, t_idx] = 1  # Marcamos la frecuencia correspondiente

        return contour_spectrogram

    # Mapeo MIDI -> bins de frecuencia en la STFT
    def midi_to_stft_bin(self, midi_note, sr, n_fft):
        """Convierte un número MIDI a un índice de bin de la STFT."""
        num_bins = n_fft // 2 + 1
        freq_hz = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))  # Convertimos MIDI a Hz
        bin_idx = int(round((freq_hz / (sr / 2)) * (n_fft // 2)))  # Convertimos Hz a índice STFT
        return np.clip(bin_idx, 0, num_bins - 1)  # Aseguramos que el índice esté en el rango
