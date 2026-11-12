"""Phase 3b — Train the ABL micro-siting PINN over the reference parcel.

Gate (blueprint Table 4): mean |continuity residual| ||R_mass||_2 < 1e-4
(velocities normalised by the geostrophic speed, lengths by the domain).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..common.fields import synthetic_dem
from ..config import CHECKPOINT_DIR, METRICS_DIR, SEED
from ..layer2_atmosphere.pinn_abl_micrositing import (ABLPhysicsLoss,
                                                      DivergenceFreeABLPINN,
                                                      hub_height_field)


def main(steps: int = 700, n_col: int = 384, n_bc: int = 160,
         seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dem = synthetic_dem(64, rng)
    dem_norm = ((dem - dem.min()) / (np.ptp(dem) + 1e-9)).astype(np.float32)
    dem_frac = 0.25 * dem_norm            # terrain occupies lower quarter of ABL

    model = DivergenceFreeABLPINN()   # continuity satisfied by construction
    # u,v normalised by |v_g|: work in nondimensional velocity space
    phys = ABLPhysicsLoss(u_geo=1.0, v_geo=0.15, lat_deg=14.2,
                          k_m=0.02, domain_m=1.0, abl_top_m=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    dem_t = torch.from_numpy(dem_frac)
    t_start = time.time()
    for step in range(steps):
        xy = torch.rand(n_col, 2)
        ij = (xy * 63).long()
        h = dem_t[ij[:, 1], ij[:, 0]]
        z = h + torch.rand(n_col) * (1.0 - h)
        col = torch.stack([xy[:, 0], xy[:, 1], z], dim=1)
        res = phys.pde_residuals(model, col)
        loss_pde = sum((r ** 2).mean() for r in res.values())

        xy_b = torch.rand(n_bc, 2)
        ij_b = (xy_b * 63).long()
        h_b = dem_t[ij_b[:, 1], ij_b[:, 0]]
        z0_b = torch.full_like(h_b, 0.01)
        bc = phys.boundary_losses(model, xy_b, h_b, z0_b)
        loss_bc = (6.0 * bc["no_slip"].mean() + 2.0 * bc["geostrophic"].mean()
                   + 1.0 * bc["log_law"].mean())
        loss = loss_pde + loss_bc
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"[L2-PINN] step {step:4d}/{steps} pde={loss_pde.item():.6f} "
                  f"bc={loss_bc.item():.6f}")

    torch.save({"model": model.state_dict()},
               CHECKPOINT_DIR / "layer2_abl_pinn.pt")

    # ---- gates ----
    xy = torch.rand(4096, 2)
    ij = (xy * 63).long()
    h = dem_t[ij[:, 1], ij[:, 0]]
    z = h + torch.rand(4096) * (1.0 - h)
    res = phys.pde_residuals(model, torch.stack([xy[:, 0], xy[:, 1], z], 1))
    mass_res = float((res["mass"] ** 2).mean().sqrt())
    momx_res = float((res["momx"] ** 2).mean().sqrt())

    hub = hub_height_field(model, dem_frac, abl_top_m=1.0)
    # 120 m / 800 m ABL — evaluate speed-up over ridge vs valley
    ridge_speed = float(np.quantile(hub["speed"], 0.95))
    valley_speed = float(np.quantile(hub["speed"], 0.05))
    speedup_pct = 100 * (ridge_speed / max(valley_speed, 1e-6) - 1)

    metrics = {
        "massResidualRms": mass_res,
        "momentumXResidualRms": momx_res,
        "ridgeVsValleySpeedupPct": round(speedup_pct, 1),
        "gate": {"massResidual<1e-4": mass_res < 1e-4},
        "trainSteps": steps,
        "trainMinutes": round((time.time() - t_start) / 60, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (METRICS_DIR / "layer2_pinn.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    main(steps=ap.parse_args().steps)
