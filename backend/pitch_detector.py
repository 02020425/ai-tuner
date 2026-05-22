"""Pitch detection and musical scale utilities."""

import numpy as np
import librosa

A4_FREQ = 440.0
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Common scales: semitone offsets from tonic
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "chromatic": list(range(12)),
}


def freq_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    return 69 + 12 * np.log2(freq / A4_FREQ)


def midi_to_freq(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return A4_FREQ * 2 ** ((midi - 69) / 12)


def hz_to_cents(freq: float, ref_freq: float) -> float:
    """Difference between two frequencies in cents."""
    return 1200 * np.log2(freq / ref_freq)


def extract_pitch(y: np.ndarray, sr: int, hop_length: int = 256) -> np.ndarray:
    """Extract pitch (F0) contour using pYIN algorithm.

    Returns:
        f0: pitch in Hz, shape (n_frames,). NaN where unvoiced.
        voiced_flag: bool array, shape (n_frames,)
        voiced_prob: float array, shape (n_frames,)
    """
    fmin = librosa.note_to_hz("E2")
    fmax = librosa.note_to_hz("C6")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    if f0 is None:
        f0 = np.full(voiced_flag.shape, np.nan)
    return f0, voiced_flag, voiced_prob


def freq_to_scale_degree(freq: float, tonic_hz: float, scale_name: str) -> int:
    """Map a frequency to the nearest scale degree (0-11)."""
    if not np.isfinite(freq) or freq <= 0:
        return -1
    midi = freq_to_midi(freq)
    tonic_midi = freq_to_midi(tonic_hz)
    semitone = int(round(midi - tonic_midi)) % 12
    return semitone


def snap_to_scale(freq: float, tonic_hz: float, scale_name: str) -> float:
    """Snap a frequency to the nearest note in the given scale."""
    if not np.isfinite(freq) or freq <= 0:
        return freq

    midi = freq_to_midi(freq)
    tonic_midi = freq_to_midi(tonic_hz)
    offset = midi - tonic_midi

    scale_semitones = SCALES.get(scale_name, SCALES["major"])
    octave = round(offset / 12)
    rel_semitone = (offset % 12 + 12) % 12

    # Find nearest scale note
    best = min(scale_semitones, key=lambda s: abs(s - rel_semitone))
    corrected_midi = tonic_midi + octave * 12 + best

    # If we're closer to the next octave's version, use that
    for oct in [octave - 1, octave + 1]:
        alt_midi = tonic_midi + oct * 12 + best
        if abs(alt_midi - midi) < abs(corrected_midi - midi):
            corrected_midi = alt_midi

    return midi_to_freq(corrected_midi)


def detect_key(y: np.ndarray, sr: int) -> tuple[str, float]:
    """Detect musical key using Krumhansl-Schmuckler profile.

    Returns:
        key_name: e.g. "C major", "A minor"
        tonic_hz: frequency of the tonic
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # 12 values

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -np.inf
    best_key = "C major"
    best_tonic = 0

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        corr_maj = np.corrcoef(rotated, major_profile)[0, 1]
        corr_min = np.corrcoef(rotated, minor_profile)[0, 1]

        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{PITCH_NAMES[i]} major"
            best_tonic = i
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{PITCH_NAMES[i]} minor"
            best_tonic = i

    tonic_hz = midi_to_freq(60 + best_tonic)
    return best_key, tonic_hz
