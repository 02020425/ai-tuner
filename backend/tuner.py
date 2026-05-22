"""Core pitch correction engine.

Three modes:
  - scale-based: snap each note to nearest scale degree (DSP)
  - reference-based: align to a reference audio, then snap to reference pitch (DSP)
  - neural: neural vocoder reconstruction with corrected pitch (requires trained model)

The DSP path uses pyrubberband. The neural path replaces the pitch-shift step
with a HiFi-GAN vocoder that generates the corrected waveform natively.
Both pipelines share the same pitch detection and correction logic.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
import torch
from dataclasses import dataclass

from pitch_detector import (
    extract_pitch,
    snap_to_scale,
    detect_key,
    hz_to_cents,
    SCALES,
)
from alignment import dtw_align
from rhythm_corrector import correct_to_reference as rhythm_to_ref, correct_to_grid as rhythm_to_grid
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Rhythm correction pre-processing
# ---------------------------------------------------------------------------

def _apply_rhythm_correction(
    y: np.ndarray,
    sr: int,
    rhythm_mode: str,
    y_ref: np.ndarray | None = None,
    sr_ref: int | None = None,
    bpm: float | None = None,
) -> np.ndarray:
    """Apply rhythm correction before pitch correction.

    Args:
        y: audio waveform
        sr: sample rate
        rhythm_mode: "none", "grid", or "reference"
        y_ref: reference audio (required for "reference" mode)
        sr_ref: reference sample rate (required for "reference" mode)
        bpm: target BPM (optional for "grid" mode, auto-detected if None)

    Returns:
        time-corrected waveform
    """
    if rhythm_mode == "none":
        return y
    elif rhythm_mode == "reference":
        if y_ref is None:
            return y
        return rhythm_to_ref(y, sr, y_ref, sr_ref)
    elif rhythm_mode == "grid":
        return rhythm_to_grid(y, sr, bpm=bpm)
    else:
        return y


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _smooth_pitch_contour(f0: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Apply median filter to remove isolated pitch jumps and vibrato overshoot.

    Only touches non-zero frames (F0 > 0 Hz or non-zero cents correction);
    zero/unvoiced frames pass through untouched.
    """
    active = np.abs(f0) > 0.1
    if not active.any():
        return f0
    smoothed = median_filter(f0, size=kernel, mode="nearest")
    smoothed[~active] = f0[~active]
    return smoothed


def _normalize_loudness(y: np.ndarray, sr: int, target_db: float = -18.0) -> np.ndarray:
    """RMS-based loudness normalization.

    Brings average loudness to target_db (LUFS-approximate), applying a
    soft limiter to prevent clipping. Preserves relative dynamics within
    the audio — this is leveling, not compression.

    Args:
        y: audio waveform
        sr: sample rate
        target_db: target RMS level in dBFS (default -18, standard for vocals)

    Returns:
        loudness-normalized audio
    """
    # Perceptual weighting: A-weighting approximates how ears hear loudness
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return y

    current_db = 20 * np.log10(rms + 1e-10)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20.0)

    # Soft limit: don't push peaks beyond -1 dBFS
    peak = np.max(np.abs(y)) * gain_linear
    if peak > 0.85:
        gain_linear = 0.85 / (np.max(np.abs(y)) + 1e-10)

    return y * gain_linear


@dataclass
class TuneResult:
    audio: np.ndarray
    sample_rate: int
    key_detected: str
    frames_processed: int
    frames_corrected: int
    avg_correction_cents: float
    method: str = "dsp"  # "dsp" or "neural"

    def __repr__(self):
        return (
            f"TuneResult(method={self.method!r}, key={self.key_detected!r}, "
            f"frames={self.frames_processed}, corrected={self.frames_corrected}, "
            f"avg_correction={self.avg_correction_cents:.1f} cents)"
        )


def tune_to_scale(
    file_path: str,
    key: str = "auto",
    scale_type: str = "major",
    correction_strength: float = 1.0,
    rhythm_mode: str = "none",
    rhythm_ref_path: str | None = None,
    rhythm_bpm: float | None = None,
) -> TuneResult:
    """Correct pitch by snapping to a musical scale.

    Args:
        file_path: path to input audio (WAV recommended)
        key: "auto" to detect, or e.g. "C", "C#", "D", ...
        scale_type: one of the keys in SCALES dict
        correction_strength: 0.0 = no correction, 1.0 = full snap to scale
        rhythm_mode: "none", "grid", or "reference"
        rhythm_ref_path: path to reference audio for rhythm alignment
        rhythm_bpm: target BPM for grid quantization (auto-detected if None)

    Returns:
        TuneResult with corrected audio and metadata.
    """
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert stereo to mono

    # Rhythm correction pre-processing
    if rhythm_mode != "none":
        y_ref = None
        sr_ref = None
        if rhythm_mode == "reference" and rhythm_ref_path:
            y_ref, sr_ref = sf.read(rhythm_ref_path)
            if y_ref.ndim > 1:
                y_ref = y_ref.mean(axis=1)
        y = _apply_rhythm_correction(y, sr, rhythm_mode, y_ref, sr_ref, rhythm_bpm)

    hop_length = 256

    # Extract pitch
    f0, voiced_flag, _ = extract_pitch(y, sr, hop_length=hop_length)

    # Detect or set key
    if key == "auto":
        key_name, tonic_hz = detect_key(y, sr)
    else:
        from pitch_detector import PITCH_NAMES, midi_to_freq

        tonic_idx = PITCH_NAMES.index(key) if key in PITCH_NAMES else 0
        tonic_hz = midi_to_freq(60 + tonic_idx)
        key_name = f"{key} {scale_type}"

    # Compute pitch correction per frame
    f0_corrected = f0.copy()
    pitch_shifts_cents = np.zeros(len(f0))

    for i in range(len(f0)):
        if voiced_flag[i] and np.isfinite(f0[i]):
            target_freq = snap_to_scale(f0[i], tonic_hz, scale_type)
            f0_corrected[i] = target_freq

    # Smooth target pitch to remove isolated jump corrections (vibrato overshoot)
    f0_corrected = _smooth_pitch_contour(f0_corrected, kernel=5)

    # Compute per-frame pitch shifts from smoothed targets
    for i in range(len(f0)):
        if voiced_flag[i] and np.isfinite(f0[i]) and f0_corrected[i] > 0:
            pitch_shifts_cents[i] = hz_to_cents(f0_corrected[i], f0[i]) * correction_strength

    # Apply pitch shifting with rubberband
    y_corrected = pyrb.pitch_shift(y, sr, pitch_shifts_cents / 100.0, hop_length=hop_length)

    # Loudness normalization
    y_corrected = _normalize_loudness(y_corrected, sr)

    frames_corrected = int((np.abs(pitch_shifts_cents) > 5.0).sum())

    return TuneResult(
        audio=y_corrected,
        sample_rate=sr,
        key_detected=key_name,
        frames_processed=len(f0),
        frames_corrected=frames_corrected,
        avg_correction_cents=float(np.mean(np.abs(pitch_shifts_cents[pitch_shifts_cents != 0]))),
        method="dsp",
    )


def tune_to_reference(
    file_path: str,
    reference_path: str,
    correction_strength: float = 1.0,
    rhythm_mode: str = "none",
) -> TuneResult:
    """Correct pitch by aligning to a reference (original singer) audio.

    Steps:
      0. (optional) Rhythm correction to reference timing
      1. Extract pitch from both input and reference
      2. DTW-align the reference pitch to input timing
      3. Snap input pitch to aligned reference pitch

    Args:
        file_path: path to the out-of-tune input audio
        reference_path: path to the reference (correct) audio
        correction_strength: 0.0 = no correction, 1.0 = full snap to reference
        rhythm_mode: "none" or "reference" (uses reference_path for rhythm alignment)

    Returns:
        TuneResult with corrected audio and metadata.
    """
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    y_ref, sr_ref = sf.read(reference_path)
    if y_ref.ndim > 1:
        y_ref = y_ref.mean(axis=1)

    # Rhythm correction pre-processing
    if rhythm_mode == "reference":
        y = _apply_rhythm_correction(y, sr, "reference", y_ref, sr_ref)

    # Resample reference to match input sample rate
    if sr_ref != sr:
        import librosa
        y_ref = librosa.resample(y_ref, orig_sr=sr_ref, target_sr=sr)

    hop_length = 256

    # Extract pitch for both
    f0_input, voiced_input, _ = extract_pitch(y, sr, hop_length=hop_length)
    f0_ref, voiced_ref, _ = extract_pitch(y_ref, sr, hop_length=hop_length)

    # Handle octave differences (simple heuristic: match median pitch)
    ref_voiced_freqs = f0_ref[np.isfinite(f0_ref) & voiced_ref]
    inp_voiced_freqs = f0_input[np.isfinite(f0_input) & voiced_input]
    if len(ref_voiced_freqs) > 0 and len(inp_voiced_freqs) > 0:
        while np.median(ref_voiced_freqs) > np.median(inp_voiced_freqs) * 1.4:
            f0_ref = f0_ref / 2
            ref_voiced_freqs = ref_voiced_freqs / 2
        while np.median(inp_voiced_freqs) > np.median(ref_voiced_freqs) * 1.4:
            f0_ref = f0_ref * 2
            ref_voiced_freqs = ref_voiced_freqs * 2

    # DTW align reference pitch to input timing
    aligned_ref_pitch = dtw_align(f0_ref, f0_input, voiced_ref, voiced_input)

    # Compute pitch shifts: snap input to aligned reference
    pitch_shifts_cents = np.zeros(len(f0_input))
    for i in range(len(f0_input)):
        if voiced_input[i] and np.isfinite(f0_input[i]) and np.isfinite(aligned_ref_pitch[i]):
            pitch_shifts_cents[i] = hz_to_cents(aligned_ref_pitch[i], f0_input[i]) * correction_strength

    # Smooth pitch shifts to avoid isolated corrections
    pitch_shifts_cents = _smooth_pitch_contour(pitch_shifts_cents, kernel=5)

    y_corrected = pyrb.pitch_shift(y, sr, pitch_shifts_cents / 100.0, hop_length=hop_length)

    # Loudness normalization
    y_corrected = _normalize_loudness(y_corrected, sr)

    frames_corrected = int((np.abs(pitch_shifts_cents) > 5.0).sum())

    return TuneResult(
        audio=y_corrected,
        sample_rate=sr,
        key_detected="reference-based",
        frames_processed=len(f0_input),
        frames_corrected=frames_corrected,
        avg_correction_cents=float(np.mean(np.abs(pitch_shifts_cents[pitch_shifts_cents != 0]))),
        method="dsp",
    )


# ---------------------------------------------------------------------------
# Neural inference path
# ---------------------------------------------------------------------------

_MODEL = None
_MODEL_PATH = None
_DEVICE = None


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_path: str | None = None) -> tuple:
    """Lazy-load the neural vocoder model.

    Searches for the model file in:
      1. model_path (if given)
      2. MODELS_DIR / "tuner.pth"  (project root models/ directory)
      3. CHECKPOINTS_DIR / "latest.pt" → extract generator weights
    """
    global _MODEL, _MODEL_PATH, _DEVICE

    project_root = Path(__file__).resolve().parent.parent
    device = _get_device()

    # Default model path
    if model_path is None:
        model_path = str(project_root / "models" / "tuner.pth")

    # If same model already loaded, return cached
    if _MODEL is not None and _MODEL_PATH == model_path and _DEVICE == device:
        return _MODEL, device

    from neural_vocoder import InferenceGenerator, load_model

    if os.path.exists(model_path):
        _MODEL = load_model(model_path, device)
    else:
        # Try loading from latest checkpoint
        ckpt_path = project_root / "checkpoints" / "latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            _MODEL = InferenceGenerator().to(device)
            _MODEL.load_state_dict(ckpt["generator"])
            _MODEL.eval()
        else:
            raise FileNotFoundError(
                f"No model found at {model_path} or {ckpt_path}. "
                "Train the model first with: python scripts/train.py"
            )

    _MODEL_PATH = model_path
    _DEVICE = device
    return _MODEL, device


def tune_neural(
    file_path: str,
    model_path: str | None = None,
    key: str = "auto",
    scale_type: str = "major",
    correction_strength: float = 1.0,
    rhythm_mode: str = "none",
    rhythm_ref_path: str | None = None,
    rhythm_bpm: float | None = None,
) -> TuneResult:
    """Neural pitch correction using a trained HiFi-GAN vocoder.

    Unlike the DSP path (which shifts pitch in frequency domain,
    causing formant distortion), this approach:
      1. Extracts mel spectrogram from the out-of-tune audio (content)
      2. Detects pitch, snaps to scale to get target pitch
      3. Feeds (mel + target_pitch) → HiFi-GAN → corrected waveform

    The vocoder generates the waveform natively at the target pitch,
    so formants stay natural regardless of correction amount.

    Args:
        file_path: path to the out-of-tune input audio
        model_path: path to trained model weights (auto-detected if None)
        key: "auto" to detect, or e.g. "C", "C#", ...
        scale_type: scale type for pitch correction
        correction_strength: 0.0 = no correction, 1.0 = full correction

    Returns:
        TuneResult with corrected audio and metadata.
    """
    model, device = _load_model(model_path)

    # Load and preprocess audio
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Rhythm correction pre-processing
    if rhythm_mode != "none":
        y_ref = None
        sr_ref = None
        if rhythm_mode == "reference" and rhythm_ref_path:
            y_ref, sr_ref = sf.read(rhythm_ref_path)
            if y_ref.ndim > 1:
                y_ref = y_ref.mean(axis=1)
        y = _apply_rhythm_correction(y, sr, rhythm_mode, y_ref, sr_ref, rhythm_bpm)

    target_sr = 22050  # Model was trained at this rate
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    hop_length = 256

    # ---- Pitch detection and correction (same logic as DSP path) ----
    f0, voiced_flag, _ = extract_pitch(y, sr, hop_length=hop_length)

    if key == "auto":
        key_name, tonic_hz = detect_key(y, sr)
    else:
        from pitch_detector import PITCH_NAMES, midi_to_freq
        tonic_idx = PITCH_NAMES.index(key) if key in PITCH_NAMES else 0
        tonic_hz = midi_to_freq(60 + tonic_idx)
        key_name = f"{key} {scale_type}"

    target_pitch = np.zeros(len(f0), dtype=np.float32)
    pitch_shifts_cents = np.zeros(len(f0))

    for i in range(len(f0)):
        if voiced_flag[i] and np.isfinite(f0[i]):
            corrected = snap_to_scale(f0[i], tonic_hz, scale_type)
            target_pitch[i] = corrected
        else:
            target_pitch[i] = 0.0

    # Smooth target pitch contour to avoid abrupt corrections
    target_pitch = _smooth_pitch_contour(target_pitch, kernel=5)

    for i in range(len(f0)):
        if voiced_flag[i] and np.isfinite(f0[i]) and target_pitch[i] > 0:
            pitch_shifts_cents[i] = hz_to_cents(target_pitch[i], f0[i]) * correction_strength

    # ---- Neural vocoder inference ----
    y_tensor = torch.from_numpy(y).float().unsqueeze(0).to(device)
    target_pitch_tensor = torch.from_numpy(target_pitch).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Extract mel from the out-of-tune audio (carries content/timbre)
        from neural_vocoder import extract_mel_torch
        mel = extract_mel_torch(y_tensor, sr, n_fft=1024, hop=hop_length, win=1024)

        # Align pitch length to mel frames
        if target_pitch_tensor.shape[1] > mel.shape[2]:
            target_pitch_tensor = target_pitch_tensor[:, :mel.shape[2]]
        elif target_pitch_tensor.shape[1] < mel.shape[2]:
            target_pitch_tensor = torch.nn.functional.pad(
                target_pitch_tensor, (0, mel.shape[2] - target_pitch_tensor.shape[1]))

        # Generate corrected waveform
        y_corrected_tensor = model(mel, target_pitch_tensor)

    y_corrected = y_corrected_tensor.squeeze().cpu().numpy()

    # Trim to expected length
    expected_len = len(y)
    if len(y_corrected) > expected_len:
        y_corrected = y_corrected[:expected_len]

    # Loudness normalization
    y_corrected = _normalize_loudness(y_corrected, sr)

    frames_corrected = int((np.abs(pitch_shifts_cents) > 5.0).sum())

    return TuneResult(
        audio=y_corrected.astype(np.float32),
        sample_rate=sr,
        key_detected=key_name,
        frames_processed=len(f0),
        frames_corrected=frames_corrected,
        avg_correction_cents=float(np.mean(np.abs(pitch_shifts_cents[pitch_shifts_cents != 0]))),
        method="neural",
    )


def tune_compare(
    file_path: str,
    model_path: str | None = None,
    key: str = "auto",
    scale_type: str = "major",
    correction_strength: float = 1.0,
    rhythm_mode: str = "none",
    rhythm_ref_path: str | None = None,
    rhythm_bpm: float | None = None,
) -> dict:
    """Run both DSP and neural correction on the same audio, return both results.

    This is the comparison endpoint — used to demonstrate the quality
    difference between DSP and AI-based correction side by side.
    """
    result_dsp = tune_to_scale(file_path, key, scale_type, correction_strength,
                                rhythm_mode, rhythm_ref_path, rhythm_bpm)
    try:
        result_neural = tune_neural(file_path, model_path, key, scale_type,
                                    correction_strength, rhythm_mode,
                                    rhythm_ref_path, rhythm_bpm)
        neural_available = True
    except (FileNotFoundError, ImportError):
        result_neural = None
        neural_available = False

    return {
        "dsp": result_dsp,
        "neural": result_neural,
        "neural_available": neural_available,
    }
