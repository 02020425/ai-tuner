"""
Lightweight HiFi-GAN inference model for neural pitch correction.

Loads weights trained by scripts/train.py and performs inference
(no discriminator, no training-only components).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal HiFi-GAN generator (same architecture as scripts/hifi_gan.py)
# ---------------------------------------------------------------------------

def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class _ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=_get_padding(kernel_size, d),
            ))

    def forward(self, x):
        for conv in self.convs:
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + residual
        return x


class _MRF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resblocks = nn.ModuleList([
            _ResBlock(channels, 3, [1, 3, 5]),
            _ResBlock(channels, 7, [1, 3, 5]),
            _ResBlock(channels, 11, [1, 3, 5]),
        ])

    def forward(self, x):
        return sum(block(x) for block in self.resblocks) / len(self.resblocks)


class InferenceGenerator(nn.Module):
    """HiFi-GAN generator for inference only (no weight norm at export time)."""

    def __init__(
        self,
        mel_bins: int = 80,
        pitch_bins: int = 1,
        h_channels: int = 512,
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        upsample_initial_channel: int = 256,
    ):
        super().__init__()
        self.mel_bins = mel_bins
        self.pitch_bins = pitch_bins
        self.input_channels = mel_bins + pitch_bins
        self.upsample_rates = upsample_rates

        self.conv_pre = nn.Conv1d(
            self.input_channels, h_channels, 7, stride=1, padding=3)

        self.upsamples = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        in_ch = h_channels
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.upsamples.append(nn.ConvTranspose1d(
                in_ch, out_ch, kernel, stride=rate,
                padding=(kernel - rate) // 2))
            self.mrfs.append(_MRF(out_ch))
            in_ch = out_ch

        self.conv_post = nn.Conv1d(in_ch, 1, 7, stride=1, padding=3)

    def forward(self, mel: torch.Tensor, pitch: torch.Tensor | None = None) -> torch.Tensor:
        if pitch is not None:
            pitch = F.interpolate(
                pitch.unsqueeze(1), size=mel.shape[2],
                mode="linear", align_corners=False)
            x = torch.cat([mel, pitch], dim=1)
        else:
            x = torch.cat([
                mel,
                torch.zeros(mel.shape[0], 1, mel.shape[2], device=mel.device),
            ], dim=1)

        x = self.conv_pre(x)
        for upsample, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            x = mrf(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def load_model(model_path: str, device: str = "cpu") -> InferenceGenerator:
    """Load a trained tuner model. Cached after first load."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    model = InferenceGenerator()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model_cache[model_path] = model
    return model


# ---------------------------------------------------------------------------
# Feature extraction for inference
# ---------------------------------------------------------------------------

def _mel_basis(n_fft: int, sr: int, n_mels: int) -> torch.Tensor:
    from librosa.filters import mel as mel_fn
    return torch.from_numpy(mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels)).float()


def extract_mel_torch(y: torch.Tensor, sr: int, n_fft: int = 1024,
                      hop: int = 256, win: int = 1024,
                      n_mels: int = 80) -> torch.Tensor:
    """Extract log-mel spectrogram. y shape: (B, T) or (T,)."""
    if y.ndim == 1:
        y = y.unsqueeze(0)
    window = torch.hann_window(win, device=y.device)
    spec = torch.stft(y, n_fft, hop, win, window, return_complex=True).abs() ** 2
    mel_b = _mel_basis(n_fft, sr, n_mels).to(y.device)
    mel = torch.matmul(mel_b, spec)
    return torch.log(torch.clamp(mel, min=1e-5))
