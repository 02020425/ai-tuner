"""Rhythm/timing correction — preprocessing before pitch correction.

Two modes:
  - reference-based: DTW-align user audio to a reference (original singer) timing
  - grid-based: detect beats/onsets, snap to a quantization grid

Runs as a pre-process step. Output is a time-corrected waveform that
then feeds into the pitch correction pipeline.
"""

import numpy as np
import librosa


def detect_beats(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Detect beat positions and BPM.

    Returns:
        beat_times: array of beat times in seconds
        bpm: estimated tempo in BPM
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if tempo is not None else 120.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times, bpm


def _dtw_time_align(y_src, y_ref, sr):
    """DTW on RMS energy envelopes to find the time-warping path.

    Returns:
        wp: (N, 2) array of (ref_frame, src_frame) alignment indices
        hop_length: used for frame indexing
    """
    hop_length = 512

    rms_src = librosa.feature.rms(y=y_src, hop_length=hop_length, frame_length=2048)[0]
    rms_ref = librosa.feature.rms(y=y_ref, hop_length=hop_length, frame_length=2048)[0]

    # Build cost matrix (cityblock on normalized envelopes)
    rms_src_norm = rms_src / (rms_src.max() + 1e-10)
    rms_ref_norm = rms_ref / (rms_ref.max() + 1e-10)

    from scipy.spatial.distance import cdist
    dist = cdist(rms_src_norm[:, None], rms_ref_norm[:, None], metric="cityblock")

    # Accumulation
    n_s, n_r = dist.shape
    D = np.full((n_s, n_r), np.inf)
    D[0, 0] = dist[0, 0]
    for i in range(1, n_s):
        D[i, 0] = D[i - 1, 0] + dist[i, 0]
    for j in range(1, n_r):
        D[0, j] = D[0, j - 1] + dist[0, j]
    for i in range(1, n_s):
        for j in range(1, n_r):
            D[i, j] = dist[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack
    i, j = n_s - 1, n_r - 1
    path = []
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    path.append((0, 0))
    path.reverse()

    return np.array(path), hop_length


def correct_to_reference(y: np.ndarray, sr: int,
                         y_ref: np.ndarray, sr_ref: int) -> np.ndarray:
    """Time-stretch user audio to match reference audio's timing via DTW.

    Each frame of the input is mapped to a corresponding frame of the
    reference, then the input is stretched/compressed to match.

    Args:
        y: user audio (out-of-time)
        sr: sample rate of y
        y_ref: reference audio (correct timing)
        sr_ref: sample rate of y_ref

    Returns:
        y_corrected: time-aligned audio at sr
    """
    if y_ref.ndim > 1:
        y_ref = y_ref.mean(axis=1)
    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr_ref != sr:
        y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=sr)
        sr_ref = sr

    # Match lengths for DTW
    max_len = max(len(y), len(y_ref))
    y_pad = np.pad(y, (0, max_len - len(y)))
    y_ref_pad = np.pad(y_ref, (0, max_len - len(y_ref)))

    path, hop_length = _dtw_time_align(y_pad, y_ref_pad, sr)

    # Build time-stretch ratio per frame based on how many ref frames map
    # to each source frame. ratio > 1 means stretch (slow down), < 1 means compress.
    src_to_ref = {}
    for ref_f, src_f in path:
        if src_f not in src_to_ref:
            src_to_ref[src_f] = []
        src_to_ref[src_f].append(ref_f)

    n_frames_src = int(np.ceil(len(y) / hop_length))
    ratios = np.ones(n_frames_src, dtype=np.float32)

    for src_f in range(n_frames_src):
        if src_f in src_to_ref:
            ref_frames = src_to_ref[src_f]
            if len(ref_frames) > 1:
                # How many ref frames per src frame? >1 means the user is
                # too fast (needs stretching), <1 means too slow.
                mapped_span = max(ref_frames) - min(ref_frames)
                ratios[src_f] = max(0.5, min(2.0, (mapped_span + 1.0)))

    # Smooth ratios to avoid artifacts
    from scipy.ndimage import uniform_filter1d
    ratios = uniform_filter1d(ratios, size=5)

    # Apply time-stretch using rubberband (small shifts, good quality)
    import pyrubberband as pyrb
    y_corrected = pyrb.time_stretch(y, sr, 1.0 / ratios, hop_length=hop_length)

    return y_corrected


def correct_to_grid(y: np.ndarray, sr: int, bpm: float | None = None,
                    grid_division: int = 16) -> np.ndarray:
    """Quantize note onsets to a metric grid.

    Detects note onsets, finds their deviation from the nearest grid
    position, and applies local time-stretch to snap them into place.

    Args:
        y: audio waveform
        sr: sample rate
        bpm: tempo; auto-detected if None
        grid_division: subdivisions per bar (16 = 16th notes)

    Returns:
        y_corrected: rhythm-quantized audio
    """
    if y.ndim > 1:
        y = y.mean(axis=1)

    if bpm is None:
        _, bpm = detect_beats(y, sr)

    beat_interval = 60.0 / bpm  # seconds per beat
    grid_interval = beat_interval / (grid_division / 4)  # seconds per grid position

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=512, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if len(onset_times) < 2:
        return y  # Nothing to quantize

    # For each onset, compute time offset from nearest grid
    hop_length = 256
    n_frames = int(np.ceil(len(y) / hop_length))
    frame_shifts = np.zeros(n_frames, dtype=np.float32)

    for onset_t in onset_times:
        nearest_grid = round(onset_t / grid_interval) * grid_interval
        offset = onset_t - nearest_grid  # positive = late, negative = early

        if abs(offset) < 0.02:  # < 20ms, not noticeable
            continue

        # Build a local stretch/compress over a window around this onset
        window_sec = 0.25  # 250ms window: stretch/compress gradually
        center_frame = int(onset_t * sr / hop_length)

        for f_offset in range(-int(window_sec * sr / hop_length),
                              int(window_sec * sr / hop_length)):
            frame_idx = center_frame + f_offset
            if 0 <= frame_idx < n_frames:
                # Triangular window: maximum correction at onset, fading out
                weight = max(0, 1.0 - abs(f_offset) * hop_length / (window_sec * sr))
                # Negative shift = speed up to correct late onset
                shift = -offset / window_sec * weight
                frame_shifts[frame_idx] += shift

    # Apply subtle time stretching
    if np.any(np.abs(frame_shifts) > 0.001):
        import pyrubberband as pyrb
        # Convert frame shifts to stretch ratio (1.0 = no change)
        stretch_ratio = 1.0 + frame_shifts
        y_corrected = pyrb.time_stretch(y, sr, stretch_ratio, hop_length=hop_length)
        # Trim/pad to original length
        if len(y_corrected) > len(y):
            y_corrected = y_corrected[:len(y)]
        elif len(y_corrected) < len(y):
            y_corrected = np.pad(y_corrected, (0, len(y) - len(y_corrected)))
        return y_corrected

    return y
