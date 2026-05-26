"""Debug: check why files are failing WORLD analysis."""
import pyworld as pw
import soundfile as sf
import numpy as np
from pathlib import Path

files = sorted(Path("data/clean").rglob("*.wav"))
idx = 6631
f = str(files[idx])
print(f"当前文件: {f}")
y, sr = sf.read(f)
if y.ndim > 1:
    y = y.mean(axis=1)
print(f"  时长: {len(y)/sr:.2f}s, 采样率: {sr}Hz")

target_sr = 22050
if sr != 22050:
    import librosa
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

y_f64 = y.astype(np.float64)
try:
    f0, t = pw.dio(y_f64, sr, f0_floor=65.0, f0_ceil=1047.0, frame_period=5.0)
    print(f"  DIO OK: {len(f0)} frames")
    f0 = pw.stonemask(y_f64, f0, t, sr)
    sp = pw.cheaptrick(y_f64, f0, t, sr)
    ap = pw.d4c(y_f64, f0, t, sr)
    print(f"  WORLD OK")
except Exception as e:
    print(f"  WORLD FAILED: {e}")
