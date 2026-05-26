"""
Generate paired training data for neural pitch correction using WORLD vocoder.

Takes clean vocal audio files and produces paired (out_of_tune, clean)
training examples by applying controlled random pitch deviations.

WORLD analysis (expensive) is done once per input file. Each training pair
randomly selects a segment, applies a single pitch shift, and synthesizes.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from tqdm import tqdm


def _shift_spectral_envelope(sp: np.ndarray, ratio: float) -> np.ndarray:
    """Shift spectral envelope by ratio via linear interpolation on frequency axis."""
    n_bins = len(sp)
    tgt_indices = np.arange(n_bins)
    src_indices = tgt_indices / ratio
    return np.interp(src_indices, tgt_indices, sp)


def _apply_pitch_shift(
    f0_seg: np.ndarray,
    sp_seg: np.ndarray,
    shift_cents: float,
    formant_shift_ratio: float,
) -> np.ndarray:
    """Apply pitch shift to copied WORLD params. Returns shifted f0."""
    f0_new = f0_seg.copy()
    if abs(shift_cents) < 0.1:
        return f0_new, sp_seg
    factor = 2.0 ** (shift_cents / 1200.0)
    voiced = f0_seg > 0
    f0_new[voiced] *= factor
    if formant_shift_ratio > 0:
        env_ratio = 1.0 + (factor - 1.0) * formant_shift_ratio
        sp_new = sp_seg.copy()
        n_bins = sp_seg.shape[1]
        tgt_indices = np.arange(n_bins)
        src_indices = tgt_indices / env_ratio
        for i in np.where(voiced)[0]:
            sp_new[i] = np.interp(src_indices, tgt_indices, sp_seg[i])
        return f0_new, sp_new
    return f0_new, sp_seg


def generate_dataset(
    input_dir: str,
    output_dir: str,
    pairs_per_file: int = 30,
    max_shift_cents: float = 300.0,
    target_sr: int = 22050,
    max_duration_sec: float = 15.0,
    formant_shift_ratio: float = 0.4,
    min_clean_ratio: float = 0.05,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    pairs_dir = output_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = ["*.wav", "*.mp3", "*.flac"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(input_path.rglob(ext)))

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
    frame_period_samples = int(target_sr * 0.005)

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

        if len(y) < sr:
            continue

        # WORLD analysis ONCE per file
        y_f64 = y.astype(np.float64)
        try:
            f0, t = pw.dio(y_f64, sr, f0_floor=65.0, f0_ceil=1047.0, frame_period=5.0)
            f0 = pw.stonemask(y_f64, f0, t, sr)
            sp = pw.cheaptrick(y_f64, f0, t, sr)
            ap = pw.d4c(y_f64, f0, t, sr)
        except Exception:
            continue

        n_frames = len(f0)
        total_samples = len(y)

        for _ in range(pairs_per_file):
            segment_dur = random.uniform(3.0, min(max_duration_sec, total_samples / sr))
            seg_samples = int(segment_dur * sr)

            if seg_samples >= total_samples:
                start_sample = 0
            else:
                start_sample = random.randint(0, total_samples - seg_samples)

            end_sample = start_sample + seg_samples
            segment_clean = y[start_sample:end_sample]

            start_frame = start_sample // frame_period_samples
            end_frame = min(n_frames, (end_sample + frame_period_samples - 1) // frame_period_samples)
            if end_frame <= start_frame:
                continue

            # Random shift: either clean or a random deviation
            if random.random() < min_clean_ratio:
                shift_cents = 0.0
            else:
                sign = random.choice([-1, 1])
                shift_cents = sign * random.uniform(20.0, max_shift_cents)

            try:
                f0_seg = f0[start_frame:end_frame]
                sp_seg = sp[start_frame:end_frame]
                ap_seg = ap[start_frame:end_frame]

                f0_shifted, sp_shifted = _apply_pitch_shift(
                    f0_seg, sp_seg, shift_cents, formant_shift_ratio)

                y_shifted = pw.synthesize(f0_shifted, sp_shifted, ap_seg, sr)
                y_shifted = y_shifted.astype(np.float32)

                # Trim to same length
                min_len = min(len(y_shifted), len(segment_clean))
                y_shifted = y_shifted[:min_len]
                y_clean = segment_clean[:min_len].astype(np.float32)

                clean_path = pairs_dir / f"{pair_idx:05d}_clean.wav"
                shifted_path = pairs_dir / f"{pair_idx:05d}_shifted.wav"
                f0_path = pairs_dir / f"{pair_idx:05d}_f0.npy"

                sf.write(str(clean_path), y_clean, sr)
                sf.write(str(shifted_path), y_shifted, sr)
                np.save(str(f0_path), f0_seg.astype(np.float32))

                metadata["pairs"].append({
                    "id": pair_idx,
                    "clean": str(clean_path.name),
                    "shifted": str(shifted_path.name),
                    "f0": str(f0_path.name),
                    "source_file": audio_file.name,
                    "duration_sec": round(min_len / sr, 2),
                })
                pair_idx += 1

            except Exception:
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
