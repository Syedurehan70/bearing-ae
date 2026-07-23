"""Autoencoder architectures.

Both are deliberately small. The point of the experiment is the evaluation
protocol, not capacity: a bottleneck that is too wide simply learns the identity
function and reconstructs faults as well as it reconstructs healthy data.
"""

from __future__ import annotations

import torch
from torch import nn


class DenseAE(nn.Module):
    """Fully-connected autoencoder for spectral inputs."""

    def __init__(self, input_dim: int, hidden: list[int], latent: int) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden:
            encoder_layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        encoder_layers.append(nn.Linear(prev, latent))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = latent
        for width in reversed(hidden):
            decoder_layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ConvAE(nn.Module):
    """1-D convolutional autoencoder for raw time windows."""

    def __init__(self, input_dim: int, channels: list[int], latent: int) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev = 1
        for width in channels:
            encoder_layers += [
                nn.Conv1d(prev, width, kernel_size=9, stride=4, padding=4),
                nn.ReLU(),
            ]
            prev = width
        self.conv = nn.Sequential(*encoder_layers)

        reduced = input_dim // (4 ** len(channels))
        self.flat_dim = prev * reduced
        self.to_latent = nn.Linear(self.flat_dim, latent)
        self.from_latent = nn.Linear(latent, self.flat_dim)
        self._shape = (prev, reduced)

        decoder_layers: list[nn.Module] = []
        widths = list(reversed(channels))
        for i, width in enumerate(widths):
            out = widths[i + 1] if i + 1 < len(widths) else 1
            decoder_layers += [
                nn.ConvTranspose1d(
                    width, out, kernel_size=9, stride=4, padding=4, output_padding=3
                )
            ]
            if i + 1 < len(widths):
                decoder_layers.append(nn.ReLU())
        self.deconv = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        z = self.to_latent(h.flatten(1))
        h = self.from_latent(z).view(-1, *self._shape)
        return self.deconv(h).squeeze(1)


def build_model(cfg: dict, input_dim: int) -> nn.Module:
    kind = cfg["model"]["kind"]
    if kind == "dense":
        return DenseAE(input_dim, cfg["model"]["hidden"], cfg["model"]["latent"])
    if kind == "conv":
        return ConvAE(input_dim, cfg["model"]["channels"], cfg["model"]["latent"])
    raise ValueError(f"unknown model kind: {kind!r}")
