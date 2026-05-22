"""Dynamic Time Warping based alignment for pitch contours."""

import numpy as np
from scipy.spatial.distance import cdist


def dtw_align(
    ref: np.ndarray,
    target: np.ndarray,
    ref_voiced: np.ndarray | None = None,
    target_voiced: np.ndarray | None = None,
) -> np.ndarray:
    """Align two pitch contours using DTW.

    Matches each frame of `target` to the best corresponding frame of `ref`,
    accounting for tempo / timing differences.

    Args:
        ref: pitch contour of reference audio (Hz), shape (n_frames_ref,)
        target: pitch contour of input audio (Hz), shape (n_frames_target,)
        ref_voiced: bool mask for voiced frames in ref
        target_voiced: bool mask for voiced frames in target

    Returns:
        aligned_ref_pitch: ref pitch mapped to target timeline, shape (n_frames_target,)
    """
    # Replace NaN with 0 for distance computation
    ref_clean = np.nan_to_num(ref, nan=0.0)
    target_clean = np.nan_to_num(target, nan=0.0)

    # Convert to MIDI for better distance metric
    def hz_to_midi_safe(freqs):
        midi = np.full_like(freqs, np.nan)
        valid = (freqs > 0) & np.isfinite(freqs)
        midi[valid] = 69 + 12 * np.log2(freqs[valid] / 440.0)
        return midi

    ref_midi = hz_to_midi_safe(ref_clean)
    target_midi = hz_to_midi_safe(target_clean)

    # Distance matrix: use voiced regions primarily
    dist = cdist(target_midi[:, None], ref_midi[:, None], metric="cityblock")
    dist = np.nan_to_num(dist, nan=12.0)  # Large distance for unvoiced

    # If voiced masks provided, penalize unvoiced matches
    if ref_voiced is not None and target_voiced is not None:
        penalty = np.ones_like(dist) * 6
        for i in range(len(target_voiced)):
            for j in range(len(ref_voiced)):
                if target_voiced[i] and ref_voiced[j]:
                    penalty[i, j] = 0
        dist = dist + penalty

    # DTW accumulation
    n_t, n_r = dist.shape
    D = np.full((n_t, n_r), np.inf)
    D[0, 0] = dist[0, 0]

    for i in range(1, n_t):
        D[i, 0] = D[i - 1, 0] + dist[i, 0]
    for j in range(1, n_r):
        D[0, j] = D[0, j - 1] + dist[0, j]

    for i in range(1, n_t):
        for j in range(1, n_r):
            D[i, j] = dist[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Backtrack
    path = []
    i, j = n_t - 1, n_r - 1
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

    # Map ref pitch to target timeline
    aligned_ref = np.full(n_t, np.nan)
    target_to_ref = {}
    for t, r in path:
        if t not in target_to_ref:
            target_to_ref[t] = r

    for t in range(n_t):
        if t in target_to_ref:
            r = target_to_ref[t]
            aligned_ref[t] = ref[r] if np.isfinite(ref[r]) else np.nan

    # Fill gaps
    last_valid = np.nan
    for t in range(n_t):
        if np.isfinite(aligned_ref[t]):
            last_valid = aligned_ref[t]
        else:
            aligned_ref[t] = last_valid

    return aligned_ref
