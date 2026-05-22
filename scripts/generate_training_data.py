"""
Generate paired training data for neural pitch correction using WORLD vocoder.

Takes clean vocal audio files and produces paired (out_of_tune, clean)
training examples by applying controlled random pitch deviations.

Unlike the previous pyrubberband approach (frequency-domain stretching),
WORLD separates audio into three independent components:
  - f0 (fundamental frequency / pitch)
  - spectral envelope (resonance / formants)
  - aperiodicity (breath / unvoiced components)

This lets us:
  1. Shift f0 independently to simulate pitch errors
  2. Optionally scale the spectral envelope along with f0 to mimic how
     real pitch errors cause formant shifts (reducing the domain gap
     between synthetic and real out-of-tune audio)
  3. Keep aperiodicity unchanged (no artifacts on breath sounds)

Usage:
    python generate_training_data.py \
        --input_dir data/clean/ \
        --output_dir data/training/ \
        --pairs_per_file 50 \
        --max_shift_cents 300
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from tqdm import tqdm


def generate_training_pair_world(
    y: np.ndarray,
    sr: int,
    max_shift_cents: float = 300.0,
    min_shift_cents: float = 20.0,
    max_segment_sec: float = 2.0,
    min_clean_ratio: float = 0.05,
    formant_shift_ratio: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create one (out_of_tune, clean) pair using WORLD vocoder.

    Divides audio into random-length segments and applies a different
    pitch shift to each via F0 manipulation. Some segments are left clean
    so the model learns when NOT to correct.

    WORLD pipeline per segment:
      1. world.analyze() → f0, spectral_envelope, aperiodicity
      2. Shift f0 by the target cents
      3. Optionally shift spectral envelope to mimic formant coupling
      4. world.synthesize() → out-of-tune waveform

    Args:
        y: clean audio waveform
        sr: sample rate
        max_shift_cents: maximum pitch deviation in cents
        min_shift_cents: minimum (below this treat as "clean")
        max_segment_sec: maximum segment duration in seconds
        min_clean_ratio: minimum fraction of audio left untouched
        formant_shift_ratio: how much the spectral envelope shifts with f0
            (0.0 = pure f0 shift, unrealistic; 0.4 = moderate coupling,
            closer to real pitch errors)

    Returns:
        y_shifted: out-of-tune version
        y_clean: original (same length)
        segments: list of {start_sample, end_sample, shift_cents} per segment
    """
    total_samples = len(y)
    max_segment_samples = int(max_segment_sec * sr)
    min_segment_samples = sr // 4

    # --- Divide into random segments with pitch shift assignments ---
    segments = []
    pos = 0
    while pos < total_samples:
        seg_len = random.randint(min_segment_samples, max_segment_samples)
        seg_len = min(seg_len, total_samples - pos)

        # Some segments stay clean so the model learns non-correction
        is_clean = random.random() < min_clean_ratio
        if is_clean:
            shift = 0.0
        else:
            sign = random.choice([-1, 1])
            magnitude = random.uniform(min_shift_cents, max_shift_cents)
            shift = sign * magnitude

        segments.append({
            "start": pos,
            "end": pos + seg_len,
            "shift_cents": shift,
        })
        pos += seg_len

    # --- WORLD analysis ---
    f0, t = pw.dio(y.astype(np.float64), sr,
                   f0_floor=65.0, f0_ceil=1047.0,
                   frame_period=5.0)
    f0 = pw.stonemask(y.astype(np.float64), f0, t, sr)
    sp = pw.cheaptrick(y.astype(np.float64), f0, t, sr)
    ap = pw.d4c(y.astype(np.float64), f0, t, sr)

    # --- Build per-frame pitch shift map ---
    n_frames = len(f0)
    frame_shifts_cents = np.zeros(n_frames, dtype=np.float64)

    for seg in segments:
        start_frame = int(seg["start"] / sr / 0.005)  # 5ms frame period
        end_frame = int(seg["end"] / sr / 0.005)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)

        if end_frame <= start_frame:
            continue

        # Smooth transitions at boundaries (5ms crossfade)
        transition = min(3, (end_frame - start_frame) // 2)
        for f in range(start_frame, end_frame):
            if transition > 0:
                if f - start_frame < transition:
                    alpha = (f - start_frame) / transition
                elif end_frame - f <= transition:
                    alpha = (end_frame - f - 1) / transition
                else:
                    alpha = 1.0
            else:
                alpha = 1.0
            frame_shifts_cents[f] = seg["shift_cents"] * alpha

    # --- Apply F0 shift ---
    f0_shifted = f0.copy()
    sp_shifted = sp.copy()

    semitone_shift = frame_shifts_cents / 100.0

    for i in range(n_frames):
        if f0[i] > 0 and abs(frame_shifts_cents[i]) > 0.1:
            # Shift F0
            f0_shifted[i] = f0[i] * (2.0 ** (semitone_shift[i] / 12.0))

            # Optionally shift spectral envelope proportionally
            # This mimics how real pitch errors couple with formant movement
            if formant_shift_ratio > 0:
                env_ratio = 1.0 + (2.0 ** (semitone_shift[i] / 12.0) - 1.0) * formant_shift_ratio
                sp_shifted[i] = _shift_spectral_envelope(sp[i], env_ratio, sr)

    # --- WORLD synthesis ---
    y_shifted = pw.synthesize(f0_shifted, sp_shifted, ap, sr)

    # --- Trim to length ---
    min_len = min(len(y_shifted), len(y))
    y_shifted = y_shifted[:min_len]
    y_clean = y[:min_len]

    return y_shifted.astype(np.float32), y_clean.astype(np.float32), segments


def _shift_spectral_envelope(sp: np.ndarray, ratio: float, sr: int) -> np.ndarray:
    """Shift the spectral envelope by a ratio (compress or expand in frequency).

    A ratio > 1.0 shifts formants upward (like higher tension),
    < 1.0 shifts them downward. Uses linear interpolation on the
    frequency axis to avoid artifacts from nearest-neighbor shifting.

    Args:
        sp: spectral envelope (n_fft_bins,), linear power
        ratio: multiplier on frequency axis (>1 shifts upward)
        sr: sample rate

    Returns:
        shifted envelope with same shape
    """
    n_bins = len(sp)
    old_indices = np.arange(n_bins, dtype=np.float64)
    new_indices = old_indices / ratio

    # Linear interpolation: for each output bin, find the corresponding
    # position in the original envelope
    shifted = np.zeros_like(sp)
    for i in range(n_bins):
        src = new_indices[i]
        lo = int(np.floor(src))
        hi = min(lo + 1, n_bins - 1)
        if lo < 0:
            shifted[i] = sp[0]
        elif lo >= n_bins - 1:
            shifted[i] = sp[-1]
        else:
            frac = src - lo
            shifted[i] = sp[lo] * (1 - frac) + sp[hi] * frac

    return shifted


def generate_dataset(
    input_dir: str,
    output_dir: str,
    pairs_per_file: int = 30,
    max_shift_cents: float = 300.0,
    target_sr: int = 22050,
    max_duration_sec: float = 15.0,
    formant_shift_ratio: float = 0.4,
):
    """Process all clean vocal files in input_dir, generate paired training data.

    Directory structure after generation:
        output_dir/
        ├── pairs/
        │   ├── 00000_clean.wav
        │   ├── 00000_shifted.wav
        │   ├── 00001_clean.wav
        │   ├── 00001_shifted.wav
        │   └── ...
        └── metadata.json
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    pairs_dir = output_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = ["*.wav", "*.mp3", "*.flac"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(input_path.glob(ext)))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        print("Place clean vocal WAV/MP3/FLAC files in data/clean/ and re-run.")
        return

    metadata = {
        "total_pairs": 0,
        "sample_rate": target_sr,
        "max_shift_cents": max_shift_cents,
        "vocoder": "WORLD",
        "formant_shift_ratio": formant_shift_ratio,
        "source_files": [str(f.name) for f in audio_files],
        "pairs": [],
    }

    pair_idx = 0

    for audio_file in tqdm(audio_files, desc="Processing files"):
        try:
            y, sr = sf.read(str(audio_file))
        except Exception as e:
            print(f"  Skipping {audio_file.name}: {e}")
            continue

        if y.ndim > 1:
            y = y.mean(axis=1)

        if sr != target_sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        if len(y) < sr:  # skip files shorter than 1 second
            continue

        for _ in range(pairs_per_file):
            segment_dur = random.uniform(3.0, min(max_duration_sec, len(y) / sr))
            segment_samples = int(segment_dur * sr)
            if segment_samples >= len(y):
                start = 0
                segment = y
            else:
                start = random.randint(0, len(y) - segment_samples)
                segment = y[start:start + segment_samples]

            try:
                y_shifted, y_clean, seg_info = generate_training_pair_world(
                    segment,
                    sr,
                    max_shift_cents=max_shift_cents,
                    formant_shift_ratio=formant_shift_ratio,
                )

                clean_path = pairs_dir / f"{pair_idx:05d}_clean.wav"
                shifted_path = pairs_dir / f"{pair_idx:05d}_shifted.wav"

                sf.write(str(clean_path), y_clean, sr)
                sf.write(str(shifted_path), y_shifted, sr)

                metadata["pairs"].append({
                    "id": pair_idx,
                    "clean": str(clean_path.name),
                    "shifted": str(shifted_path.name),
                    "source_file": audio_file.name,
                    "duration_sec": round(len(y_clean) / sr, 2),
                    "segments": seg_info,
                })
                pair_idx += 1

            except Exception as e:
                continue

    metadata["total_pairs"] = pair_idx

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {pair_idx} training pairs in {output_path}")
    print(f"  Clean audio  → {pairs_dir}/XXXXX_clean.wav")
    print(f"  Shifted audio → {pairs_dir}/XXXXX_shifted.wav")
    print(f"  Metadata     → {output_path}/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data for neural tuner (WORLD-based)"
    )
    parser.add_argument("--input_dir", default="data/clean/",
                        help="Directory of clean vocal audio files")
    parser.add_argument("--output_dir", default="data/training/",
                        help="Output directory for paired data")
    parser.add_argument("--pairs_per_file", type=int, default=30,
                        help="Number of variations per input file")
    parser.add_argument("--max_shift_cents", type=float, default=300.0,
                        help="Maximum pitch deviation in cents")
    parser.add_argument("--target_sr", type=int, default=22050,
                        help="Target sample rate")
    parser.add_argument("--max_duration_sec", type=float, default=15.0,
                        help="Maximum segment duration in seconds")
    parser.add_argument("--formant_shift_ratio", type=float, default=0.4,
                        help="Formant coupling ratio (0.0=pure f0 shift, "
                             "0.4=moderate, 1.0=full coupling)")
    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pairs_per_file=args.pairs_per_file,
        max_shift_cents=args.max_shift_cents,
        target_sr=args.target_sr,
        max_duration_sec=args.max_duration_sec,
        formant_shift_ratio=args.formant_shift_ratio,
    )
