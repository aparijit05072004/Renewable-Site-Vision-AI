"""Stage 2B — NVIDIA CorrDiff generative downscaling, 25 km -> 2 km
(blueprint §6.2, reference implementation §6.3).

Two decoupled steps, exactly as in the blueprint:
  Step 1  Deterministic regression U-Net mu_0 predicting the conditional
          mean of the 2 km field from the upsampled coarse patch + static
          topography:  L_reg = || y_2km - mu_0(x_c, z_topo) ||^2
  Step 2  EDM score-matching diffusion U-Net D_phi synthesising the
          stochastic fine-scale residual r = y_2km - mu_0, restoring the
          k^-5/3 extreme-turbulence spectrum that MSE training smooths out.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU())

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    """3-level U-Net used for both the regression mean and the EDM denoiser."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 40,
                 sigma_conditioned: bool = False):
        super().__init__()
        self.sigma_conditioned = sigma_conditioned
        extra = 1 if sigma_conditioned else 0
        self.enc1 = ConvBlock(in_ch + extra, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor,
                sigma: torch.Tensor | None = None) -> torch.Tensor:
        if self.sigma_conditioned:
            smap = torch.log(sigma.view(-1, 1, 1, 1) / 0.5).expand(
                -1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, smap], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class CorrDiffCascadedDownscaler(nn.Module):
    """Two-step corrective diffusion (25 km -> 2 km)."""

    def __init__(self, channels: int = 6, topo_ch: int = 2, base: int = 40):
        super().__init__()
        self.channels = channels
        self.regression_unet = SmallUNet(channels + topo_ch, channels, base)
        self.diffusion_unet = SmallUNet(channels + channels + topo_ch, channels,
                                        base, sigma_conditioned=True)

    # ------------------------- losses (EDM, Karras et al.) -------------
    def regression_loss(self, x_coarse_up, z_topo, y_target) -> torch.Tensor:
        pred = self.regression_unet(torch.cat([x_coarse_up, z_topo], 1))
        return F.mse_loss(pred, y_target)

    def diffusion_loss(self, x_coarse_up, z_topo, y_target,
                       p_mean: float = -1.2, p_std: float = 1.2,
                       sigma_data: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            y_mean = self.regression_unet(torch.cat([x_coarse_up, z_topo], 1))
            residual = y_target - y_mean
        b = residual.shape[0]
        sigma = (torch.randn(b, 1, 1, 1, device=residual.device)
                 * p_std + p_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / ((sigma * sigma_data) ** 2)
        noisy = residual + torch.randn_like(residual) * sigma
        denoised = self._edm_denoise(noisy, sigma, x_coarse_up, z_topo,
                                     sigma_data)
        return (weight * (denoised - residual) ** 2).mean()

    def _edm_denoise(self, noisy, sigma, x_coarse_up, z_topo, sigma_data=0.5):
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        net_in = torch.cat([c_in * noisy, x_coarse_up, z_topo], 1)
        raw = self.diffusion_unet(net_in, sigma.squeeze())
        return c_skip * noisy + c_out * raw

    # ------------------------- sampling --------------------------------
    @torch.no_grad()
    def sample(self, x_coarse_up: torch.Tensor, z_topo: torch.Tensor,
               n_steps: int = 18, sigma_min: float = 0.002,
               sigma_max: float = 6.0, rho: float = 7.0,
               sigma_data: float = 0.5) -> torch.Tensor:
        """Karras EDM deterministic (Heun) sampler for the residual,
        added onto the deterministic U-Net mean."""
        y_mean = self.regression_unet(torch.cat([x_coarse_up, z_topo], 1))
        steps = torch.arange(n_steps, device=x_coarse_up.device)
        sigmas = (sigma_max ** (1 / rho) + steps / (n_steps - 1)
                  * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        x = torch.randn_like(y_mean) * sigmas[0]
        b = x.shape[0]
        for i in range(n_steps):
            s_cur = sigmas[i].expand(b).view(-1, 1, 1, 1)
            s_next = sigmas[i + 1].expand(b).view(-1, 1, 1, 1)
            denoised = self._edm_denoise(x, s_cur, x_coarse_up, z_topo, sigma_data)
            d = (x - denoised) / s_cur
            x_next = x + (s_next - s_cur) * d
            if i < n_steps - 1:  # Heun correction
                denoised2 = self._edm_denoise(x_next, s_next, x_coarse_up,
                                              z_topo, sigma_data)
                d2 = (x_next - denoised2) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
            x = x_next
        return y_mean + x
