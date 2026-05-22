"""
Generate paired training data for neural pitch correction.

Takes a directory of clean vocal audio files and produces paired
(out_of_tune, clean) training examples by applying controlled random
pitch deviations with DSP.

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
import pyrubberband as pyrb
import soundfile as sf
from tqdm import tqdm


def generate_training_pair(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    max_shift_cents: float = 300.0,
    min_shift_cents: float = 20.0,
    max_segment_sec: float = 2.0,
    min_clean_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create one (out_of_tune, clean) pair by applying segment-level pitch shifts.

    Divides the audio into random-length segments and applies a different
    pitch shift amount to each. Some segments are left clean (no shift)
    so the model learns when NOT to correct.

    Args:
        y: clean audio waveform
        sr: sample rate
        hop_length: hop size for segment division
        max_shift_cents: maximum pitch deviation in cents
        min_shift_cents: minimum (below this, treat as "clean")
        max_segment_sec: maximum segment duration in seconds
        min_clean_ratio: minimum fraction of audio left untouched

    Returns:
        y_shifted: out-of-tune version
        y_clean: original (same length)
        segments: list of {start_sample, end_sample, shift_cents} for each segment
    """
    total_samples = len(y)
    max_segment_samples = int(max_segment_sec * sr)
    min_segment_samples = sr // 4  # ~0.25s minimum

    # Divide into random segments
    segments = []
    pos = 0
    while pos < total_samples:
        seg_len = random.randint(min_segment_samples, max_segment_samples)
        seg_len = min(seg_len, total_samples - pos)

        # Decide if this segment should stay clean
        is_clean = random.random() < min_clean_ratio
        if is_clean:
            shift = 0.0
        else:
            # Random shift: could be flat or sharp
            sign = random.choice([-1, 1])
            magnitude = random.uniform(min_shift_cents, max_shift_cents)
            shift = sign * magnitude

        segments.append({
            "start": pos,
            "end": pos + seg_len,
            "shift_cents": shift,
        })
        pos += seg_len

    # Build per-frame pitch shift map
    n_frames = (total_samples + hop_length - 1) // hop_length
    frame_shifts = np.zeros(n_frames, dtype=np.float32)

    for seg in segments:
        start_frame = seg["start"] // hop_length
        end_frame = (seg["end"] + hop_length - 1) // hop_length
        end_frame = min(end_frame, n_frames)
        # Smooth transition at segment boundaries
        transition_frames = min(3, (end_frame - start_frame) // 2)
        for f in range(start_frame, end_frame):
            if transition_frames > 0:
                if f - start_frame < transition_frames:
                    alpha = (f - start_frame) / transition_frames
                elif end_frame - f <= transition_frames:
                    alpha = (end_frame - f - 1) / transition_frames
                else:
                    alpha = 1.0
            else:
                alpha = 1.0
            frame_shifts[f] = seg["shift_cents"] * alpha

    # Apply pitch shift using rubberband
    semitone_shift = frame_shifts / 100.0  # cents → semitones
    y_shifted = pyrb.pitch_shift(y, sr, semitone_shift, hop_length=hop_length)

    # Trim to same length
    min_len = min(len(y_shifted), len(y))
    y_shifted = y_shifted[:min_len]
    y_clean = y[:min_len]

    return y_shifted, y_clean, segments


def generate_dataset(
    input_dir: str,
    output_dir: str,
    pairs_per_file: int = 30,
    max_shift_cents: float = 300.0,
    target_sr: int = 22050,
    max_duration_sec: float = 15.0,
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

    audio_files = list(input_path.glob("*.wav")) + \
                  list(input_path.glob("*.mp3")) + \
                  list(input_path.glob("*.flac"))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        print("Place clean vocal WAV/MP3/FLAC files in data/clean/ and re-run.")
        return

    metadata = {
        "total_pairs": 0,
        "sample_rate": target_sr,
        "max_shift_cents": max_shift_cents,
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

        # Resample if needed
        if sr != target_sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Trim to max duration
        max_samples = int(max_duration_sec * sr)
        if len(y) < sr:  # skip files shorter than 1 second
            continue

        for _ in range(pairs_per_file):
            # Take a random segment of the file
            segment_dur = random.uniform(3.0, min(max_duration_sec, len(y) / sr))
            segment_samples = int(segment_dur * sr)
            if segment_samples >= len(y):
                start = 0
                segment = y
            else:
                start = random.randint(0, len(y) - segment_samples)
                segment = y[start:start + segment_samples]

            try:
                y_shifted, y_clean, seg_info = generate_training_pair(
                    segment,
                    sr,
                    max_shift_cents=max_shift_cents,
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
    print(f"  Clean audio → {pairs_dir}/XXXXX_clean.wav")
    print(f"  Shifted audio → {pairs_dir}/XXXXX_shifted.wav")
    print(f"  Metadata → {output_path}/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for neural tuner")
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
    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pairs_per_file=args.pairs_per_file,
        max_shift_cents=args.max_shift_cents,
        target_sr=args.target_sr,
        max_duration_sec=args.max_duration_sec,
    )
