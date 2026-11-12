"""Stage 2C — Sub-kilometre micro-siting via Physics-Informed Neural
Networks (blueprint §6.3).

ABL-PINN  f_psi(x, y, z) -> (u, v, w, p)  enforcing over the parcel:
  * incompressible continuity        du/dx + dv/dy + dw/dz = 0
  * steady Navier-Stokes / Ekman x:  u u_x + v u_y + w u_z - f(v - v_g)
                                       + (1/rho) p_x - K_m lap(u) = 0
  * steady Navier-Stokes / Ekman y:  u v_x + v v_y + w v_z + f(u - u_g)
                                       + (1/rho) p_y - K_m lap(v) = 0
  * log-law no-slip boundary at z = h_DEM(x,y) + z0_image(x,y)
  * geostrophic recovery at the ABL top

Coordinates are normalised to the parcel box; the DEM surface and the
Layer-0 pixel roughness z0(x,y) condition the lower boundary.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

OMEGA_EARTH = 7.2921e-5
VON_KARMAN = 0.41


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 3, n_freq: int = 24, scale: float = 3.0):
        super().__init__()
        b = torch.randn(in_dim, n_freq) * scale
        self.register_buffer("B", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class ABLMicroSitingPINN(nn.Module):
    """MLP with Fourier features mapping (x,y,z) -> (u,v,w,p)."""

    def __init__(self, hidden: int = 96, layers: int = 5, n_freq: int = 24):
        super().__init__()
        self.ff = FourierFeatures(3, n_freq)
        dims = [3 + 2 * n_freq] + [hidden] * layers + [4]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(self.ff(xyz))


class DivergenceFreeABLPINN(nn.Module):
    """Exactly incompressible ABL-PINN via vector-potential parametrisation.

    The network emits a vector potential A(x,y,z) and pressure p; velocity is
    its curl, u = nabla x A, so the continuity equation du/dx+dv/dy+dw/dz = 0
    is satisfied *identically* (to autograd precision ~1e-6) rather than as a
    soft penalty — the standard hard-constraint construction for
    incompressible PINNs.  Momentum and boundary conditions are still learned.
    """

    def __init__(self, hidden: int = 96, layers: int = 5, n_freq: int = 20):
        super().__init__()
        self.ff = FourierFeatures(3, n_freq)
        dims = [3 + 2 * n_freq] + [hidden] * layers + [4]   # A1, A2, A3, p
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return (u, v, w, p) with u = curl(A); differentiable in xyz."""
        if not xyz.requires_grad:
            xyz = xyz.requires_grad_(True)
        out = self.net(self.ff(xyz))
        a1, a2, a3, p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        ones = torch.ones_like(a1)
        g1 = torch.autograd.grad(a1, xyz, ones, create_graph=True)[0]
        g2 = torch.autograd.grad(a2, xyz, ones, create_graph=True)[0]
        g3 = torch.autograd.grad(a3, xyz, ones, create_graph=True)[0]
        u = g3[:, 1] - g2[:, 2]      # dA3/dy - dA2/dz
        v = g1[:, 2] - g3[:, 0]      # dA1/dz - dA3/dx
        w = g2[:, 0] - g1[:, 1]      # dA2/dx - dA1/dy
        return torch.stack([u, v, w, p], dim=1)


class ABLPhysicsLoss:
    """PDE + boundary residuals for the ABL-PINN."""

    def __init__(self, u_geo: float, v_geo: float, lat_deg: float,
                 rho: float = 1.12, k_m: float = 8.0,
                 domain_m: float = 2000.0, abl_top_m: float = 800.0):
        self.ug, self.vg = u_geo, v_geo
        self.f_cor = 2 * OMEGA_EARTH * math.sin(math.radians(lat_deg))
        self.rho = rho
        self.k_m = k_m
        self.L = domain_m           # horizontal scale (normalised coords * L)
        self.H = abl_top_m          # vertical scale

    # ------------------------------------------------------------------
    def _grads(self, out: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(out, xyz, torch.ones_like(out),
                                   create_graph=True)[0]

    def pde_residuals(self, model: nn.Module, xyz: torch.Tensor
                      ) -> dict[str, torch.Tensor]:
        xyz = xyz.requires_grad_(True)
        uvwp = model(xyz)
        u, v, w, p = uvwp[:, 0], uvwp[:, 1], uvwp[:, 2], uvwp[:, 3]
        gu = self._grads(u, xyz) / torch.tensor([self.L, self.L, self.H])
        gv = self._grads(v, xyz) / torch.tensor([self.L, self.L, self.H])
        gw = self._grads(w, xyz) / torch.tensor([self.L, self.L, self.H])
        gp = self._grads(p, xyz) / torch.tensor([self.L, self.L, self.H])

        mass = gu[:, 0] + gv[:, 1] + gw[:, 2]

        # Laplacians via second autograd pass on first derivatives
        lap_u = sum(self._grads(gu[:, i], xyz)[:, i]
                    / (self.L if i < 2 else self.H) for i in range(3))
        lap_v = sum(self._grads(gv[:, i], xyz)[:, i]
                    / (self.L if i < 2 else self.H) for i in range(3))

        adv_u = u * gu[:, 0] + v * gu[:, 1] + w * gu[:, 2]
        adv_v = u * gv[:, 0] + v * gv[:, 1] + w * gv[:, 2]
        mom_x = adv_u - self.f_cor * (v - self.vg) + gp[:, 0] / self.rho \
            - self.k_m * lap_u
        mom_y = adv_v + self.f_cor * (u - self.ug) + gp[:, 1] / self.rho \
            - self.k_m * lap_v
        return {"mass": mass, "momx": mom_x, "momy": mom_y}

    # ------------------------------------------------------------------
    def boundary_losses(self, model: nn.Module, xy: torch.Tensor,
                        h_dem_norm: torch.Tensor, z0_norm: torch.Tensor
                        ) -> dict[str, torch.Tensor]:
        """No-slip at terrain surface; geostrophic recovery aloft; log-law
        profile anchoring in the surface layer."""
        surf = torch.cat([xy, (h_dem_norm + z0_norm).unsqueeze(1)], dim=1)
        uvwp_s = model(surf)
        no_slip = (uvwp_s[:, :3] ** 2).sum(dim=1)

        top = torch.cat([xy, torch.ones_like(h_dem_norm).unsqueeze(1)], dim=1)
        uvwp_t = model(top)
        geo = (uvwp_t[:, 0] - self.ug) ** 2 + (uvwp_t[:, 1] - self.vg) ** 2 \
            + uvwp_t[:, 2] ** 2

        # log-law anchor at z = 120 m above local terrain
        z_hub = h_dem_norm + 120.0 / self.H
        hub = torch.cat([xy, z_hub.unsqueeze(1)], dim=1)
        uvwp_h = model(hub)
        g_mag = math.hypot(self.ug, self.vg)
        ustar = VON_KARMAN * g_mag / math.log(self.H / 0.1)
        target = (ustar / VON_KARMAN) * torch.log(
            torch.tensor(120.0) / (z0_norm * self.H).clamp(1e-4))
        target = target.clamp(0.0, 1.35 * g_mag)
        speed_h = torch.sqrt(uvwp_h[:, 0] ** 2 + uvwp_h[:, 1] ** 2 + 1e-9)
        log_law = (speed_h - target) ** 2
        return {"no_slip": no_slip, "geostrophic": geo, "log_law": log_law}


def hub_height_field(model: nn.Module, dem_norm: np.ndarray,
                     abl_top_m: float = 800.0, grid: int = 64
                     ) -> dict[str, np.ndarray]:
    """Evaluate (u, v, w, p) at 120 m hub height over the parcel grid.
    (Runs with autograd enabled: the divergence-free model differentiates
    its vector potential to produce velocities.)"""
    n = grid
    xs = torch.linspace(0, 1, n)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    h = torch.from_numpy(
        np.asarray(dem_norm, dtype=np.float32))
    if h.shape != (n, n):
        h = torch.nn.functional.interpolate(
            h[None, None], (n, n), mode="bilinear", align_corners=False)[0, 0]
    z = h + 120.0 / abl_top_m
    pts = torch.stack([xx.flatten(), yy.flatten(), z.flatten()], dim=1)
    uvwp = model(pts).reshape(n, n, 4).detach().cpu().numpy()
    return {"u": uvwp[..., 0], "v": uvwp[..., 1],
            "w": uvwp[..., 2], "p": uvwp[..., 3],
            "speed": np.hypot(uvwp[..., 0], uvwp[..., 1])}
