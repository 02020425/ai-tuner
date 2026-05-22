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
) -> TuneResult:
    """Correct pitch by snapping to a musical scale.

    Args:
        file_path: path to input audio (WAV recommended)
        key: "auto" to detect, or e.g. "C", "C#", "D", ...
        scale_type: one of the keys in SCALES dict
        correction_strength: 0.0 = no correction, 1.0 = full snap to scale

    Returns:
        TuneResult with corrected audio and metadata.
    """
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert stereo to mono

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
            shift = hz_to_cents(target_freq, f0[i]) * correction_strength
            pitch_shifts_cents[i] = shift

    # Apply pitch shifting with rubberband
    shift_map = [(t * hop_length / sr, shift) for t, shift in enumerate(pitch_shifts_cents)]
    y_corrected = pyrb.pitch_shift(y, sr, pitch_shifts_cents / 100.0, hop_length=hop_length)

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
) -> TuneResult:
    """Correct pitch by aligning to a reference (original singer) audio.

    Steps:
      1. Extract pitch from both input and reference
      2. DTW-align the reference pitch to input timing
      3. Snap input pitch to aligned reference pitch

    Args:
        file_path: path to the out-of-tune input audio
        reference_path: path to the reference (correct) audio
        correction_strength: 0.0 = no correction, 1.0 = full snap to reference

    Returns:
        TuneResult with corrected audio and metadata.
    """
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    y_ref, sr_ref = sf.read(reference_path)
    if y_ref.ndim > 1:
        y_ref = y_ref.mean(axis=1)

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
            cents = hz_to_cents(aligned_ref_pitch[i], f0_input[i])
            pitch_shifts_cents[i] = cents * correction_strength

    y_corrected = pyrb.pitch_shift(y, sr, pitch_shifts_cents / 100.0, hop_length=hop_length)

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
            pitch_shifts_cents[i] = hz_to_cents(corrected, f0[i]) * correction_strength
        else:
            target_pitch[i] = 0.0

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
) -> dict:
    """Run both DSP and neural correction on the same audio, return both results.

    This is the comparison endpoint — used to demonstrate the quality
    difference between DSP and AI-based correction side by side.
    """
    result_dsp = tune_to_scale(file_path, key, scale_type, correction_strength)
    try:
        result_neural = tune_neural(file_path, model_path, key, scale_type, correction_strength)
        neural_available = True
    except (FileNotFoundError, ImportError):
        result_neural = None
        neural_available = False

    return {
        "dsp": result_dsp,
        "neural": result_neural,
        "neural_available": neural_available,
    }
