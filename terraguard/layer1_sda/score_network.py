"""Denoising score network for atmospheric-state diffusion (Layer 1).

A compact time-conditioned U-Net s_theta(x_t, t) ~ grad_x log p_t(x_t)
trained with epsilon-prediction DDPM objective on regional atmospheric
tensors [C=6, 32, 32].  Serves as the unconditional prior for the
guided reverse-time SDA in ``sda_latent_assimilator``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ScoreUNet(nn.Module):
    def __init__(self, channels: int = 6, base: int = 48, t_dim: int = 128):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(),
                                   nn.Linear(t_dim, t_dim))
        self.inc = nn.Conv2d(channels, base, 3, padding=1)
        self.d1 = ResBlock(base, base, t_dim)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.d2 = ResBlock(base * 2, base * 2, t_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = ResBlock(base * 4, base * 4, t_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.u2 = ResBlock(base * 4, base * 2, t_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.u1 = ResBlock(base * 2, base, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                 nn.Conv2d(base, channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))
        h0 = self.inc(x)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        hm = self.mid(self.down2(h2), temb)
        u2 = self.u2(torch.cat([self.up2(hm), h2], 1), temb)
        u1 = self.u1(torch.cat([self.up1(u2), h1], 1), temb)
        return self.out(u1)


def cosine_alpha_bars(steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine noise schedule -> cumulative alpha_bar[t]."""
    t = torch.linspace(0, steps, steps + 1)
    f = torch.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return ab[1:].clamp(1e-5, 0.9999)
