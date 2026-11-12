"""Score-Based Data Assimilation via guided reverse-time diffusion
(blueprint §5, reference implementation §5.1).

Fuses sparse NIWE 120 m Wind-Atlas / IoT mast observations into coarse
ERA5/GraphCast atmospheric tensors.  Bayes' rule in score space:

    grad log p_t(x_t | y) = s_theta(x_t, t)
        - gamma * grad_x [ 1/(2 sigma_t^2) (y - H(x0_hat))^T Sigma_y^-1 (y - H(x0_hat)) ]

with x0_hat the denoised Tweedie estimate of the clean state.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreBasedDataAssimilator:
    """Guided reverse-time diffusion over atmospheric state tensors."""

    def __init__(self, score_model: nn.Module, obs_precision: torch.Tensor,
                 device: str = "cpu"):
        self.score_model = score_model.to(device).eval()
        self.obs_precision = obs_precision.to(device)   # [C, C] per-channel precision
        self.device = device

    # ------------------------------------------------------------------
    @staticmethod
    def observation_operator(x_state: torch.Tensor,
                             station_coords: torch.Tensor) -> torch.Tensor:
        """Bilinear sampling of the state at station locations.

        x_state: [B, C, H, W]; station_coords: [B, N, 2] in [-1, 1] grid coords.
        Returns [B, N, C].
        """
        grid = station_coords[:, :, None, :]            # [B, N, 1, 2]
        sampled = F.grid_sample(x_state, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return sampled.squeeze(-1).permute(0, 2, 1)     # [B, N, C]

    # ------------------------------------------------------------------
    def assimilate(self, x_coarse_prior: torch.Tensor, y_obs: torch.Tensor,
                   station_coords: torch.Tensor, alpha_bars: torch.Tensor,
                   guidance_scale: float = 1.6,
                   start_frac: float = 0.35) -> torch.Tensor:
        """Anchor the coarse prior to empirical mast observations.

        Rather than sampling from pure noise, the prior field is forward-
        diffused to an intermediate step (start_frac of the schedule) and
        then integrated back with likelihood guidance — preserving the
        synoptic structure while correcting local bias (SDEdit-style
        posterior anchoring).
        """
        ab = alpha_bars.to(self.device)
        n_steps = len(ab)
        t_start = max(1, int(n_steps * start_frac))
        x_prior = x_coarse_prior.to(self.device)
        y = y_obs.to(self.device)
        coords = station_coords.to(self.device)

        noise = torch.randn_like(x_prior)
        x = (ab[t_start - 1].sqrt() * x_prior
             + (1 - ab[t_start - 1]).sqrt() * noise)

        for t in reversed(range(t_start)):
            ab_t = ab[t]
            ab_prev = ab[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            t_batch = torch.full((x.shape[0],), t, device=self.device,
                                 dtype=torch.long)
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                eps = self.score_model(x_in, t_batch)
                x0_hat = (x_in - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                y_pred = self.observation_operator(x0_hat, coords)
                residual = y - y_pred                                    # [B,N,C]
                energy = 0.5 * torch.einsum("bnc,cd,bnd->", residual,
                                            self.obs_precision, residual)
                obs_grad = torch.autograd.grad(energy, x_in)[0]

            eps_guided = eps + guidance_scale * (1 - ab_t).sqrt() * obs_grad
            x0_guided = (x - (1 - ab_t).sqrt() * eps_guided) / ab_t.sqrt()
            # DDIM deterministic update
            x = ab_prev.sqrt() * x0_guided + (1 - ab_prev).sqrt() * eps_guided
            x = x.detach()
        return x
