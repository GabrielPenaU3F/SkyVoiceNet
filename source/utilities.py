import librosa
import numpy as np
import torch
import sounddevice as sd
from matplotlib import pyplot as plt


def convert_to_midi_note(freq_tensor):
    midi_number = 69 + 12 * torch.log2(freq_tensor / 440)
    return torch.clamp(torch.round(midi_number), 0, 127).to(torch.int)

def midi_to_frequency(midi_number):
    """Convert MIDI number to frequency."""
    return 440.0 * (2.0 ** ((midi_number - 69) / 12.0))

def series_to_tensor(series):
    return torch.tensor(np.stack(series)[:, None, :, :], dtype=torch.float32)

def play_midi_notes(midi_numbers, duration=0.1, sample_rate=44100):
    """
    Play a sequence of MIDI notes.
    Args:
        midi_numbers: Array of MIDI note numbers.
        duration: Duration of each note in seconds.
        sample_rate: Audio sample rate in Hz.
    """
    audio = []
    for midi_number in midi_numbers:
        if midi_number < 0 or midi_number > 127:
            # Skip invalid MIDI notes
            continue
        frequency = midi_to_frequency(midi_number)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Generate a sine wave for the given frequency
        note = 0.5 * np.sin(2 * np.pi * frequency * t)
        audio.append(note)
    # Concatenate all notes and normalize
    audio = np.concatenate(audio).astype(np.float32)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

def draw_spectrograms(spectrogram_1, spectrogram_2, title_1='', title_2=''):
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    im_1 = ax[0].imshow(spectrogram_1, aspect="auto", origin='lower', cmap='magma')
    ax[0].set_title(title_1)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Freq")
    fig.colorbar(im_1, format='%+2.0f ', label="Intensity")

    im_2 = ax[1].imshow(spectrogram_2, aspect="auto", origin='lower', cmap='magma')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Freq")
    ax[1].set_title(title_2)
    fig.colorbar(im_2, label="Intensity")

    fig.tight_layout()
    plt.show()

def draw_single_spectrogram(spectrogram, title=''):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(spectrogram, aspect="auto", origin='lower', cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq")
    fig.colorbar(im, label="Intensity")

    fig.tight_layout()
    plt.show()
