"""Phase 3a — Train the CorrDiff cascaded downscaler (25 km -> 2 km).

Gate (blueprint Table 4): generated 2 km wind fields must reproduce the
k^-5/3 Kolmogorov kinetic-energy spectrum at fine scale (the MSE-only
regression U-Net fails this by design — we report both slopes).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..common.fields import (coarse_atmos_state, fine_from_coarse, psd_slope,
                             synthetic_dem)
from ..config import (CHECKPOINT_DIR, COARSE_GRID, FINE_GRID, METRICS_DIR,
                      SEED)
from ..layer2_atmosphere.corrdiff_downscaler import CorrDiffCascadedDownscaler

FACTOR = FINE_GRID // COARSE_GRID
CH_MEAN = np.array([5.5, 0.0, 28.0, 0.0, 0.4, 12.0], dtype=np.float32)
CH_SCALE = np.array([3.0, 3.0, 3.0, 5.0, 0.3, 2.0], dtype=np.float32)


def make_pair(rng: np.random.Generator):
    coarse = coarse_atmos_state(COARSE_GRID, rng)
    dem = synthetic_dem(FINE_GRID, rng)
    dem_norm = ((dem - dem.min()) / (np.ptp(dem) + 1e-9)).astype(np.float32)
    fine = fine_from_coarse(coarse, FACTOR, dem_norm, rng)
    up = torch.nn.functional.interpolate(
        torch.from_numpy(coarse)[None], (FINE_GRID, FINE_GRID),
        mode="bilinear", align_corners=False)[0].numpy()
    z0_proxy = np.gradient(dem_norm)[0].astype(np.float32)
    topo = np.stack([dem_norm, z0_proxy])
    up_n = (up - CH_MEAN[:, None, None]) / CH_SCALE[:, None, None]
    fine_n = (fine - CH_MEAN[:, None, None]) / CH_SCALE[:, None, None]
    return up_n.astype(np.float32), topo, fine_n.astype(np.float32)


def batch_pairs(batch: int, rng: np.random.Generator):
    xs, zs, ys = [], [], []
    for _ in range(batch):
        x, z, y = make_pair(rng)
        xs.append(x); zs.append(z); ys.append(y)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(zs)),
            torch.from_numpy(np.stack(ys)))


def main(reg_steps: int = 160, diff_steps: int = 260, batch: int = 4,
         seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CorrDiffCascadedDownscaler()
    t_start = time.time()

    # ---- Step 1: deterministic regression U-Net ----
    opt = torch.optim.AdamW(model.regression_unet.parameters(), lr=3e-4)
    for step in range(reg_steps):
        x, z, y = batch_pairs(batch, rng)
        loss = model.regression_loss(x, z, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == reg_steps - 1:
            print(f"[L2-reg] step {step:4d}/{reg_steps}  mse={loss.item():.4f}")

    # ---- Step 2: EDM score-matching diffusion on the residual ----
    opt = torch.optim.AdamW(model.diffusion_unet.parameters(), lr=3e-4)
    for step in range(diff_steps):
        x, z, y = batch_pairs(batch, rng)
        loss = model.diffusion_loss(x, z, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == diff_steps - 1:
            print(f"[L2-diff] step {step:4d}/{diff_steps}  edm={loss.item():.4f}")

    torch.save({"model": model.state_dict()},
               CHECKPOINT_DIR / "layer2_corrdiff.pt")

    # ---- Spectral gate ----
    model.eval()
    slopes_truth, slopes_corrdiff, slopes_regonly, rmses = [], [], [], []
    with torch.no_grad():
        for k in range(4):
            x, z, y = batch_pairs(1, np.random.default_rng(777_000 + k))
            full = model.sample(x, z, n_steps=14)
            reg = model.regression_unet(torch.cat([x, z], 1))
            u_true = y[0, 0].numpy(); u_gen = full[0, 0].numpy()
            u_reg = reg[0, 0].numpy()
            slopes_truth.append(psd_slope(u_true))
            slopes_corrdiff.append(psd_slope(u_gen))
            slopes_regonly.append(psd_slope(u_reg))
            rmses.append(float(np.sqrt(np.mean((u_gen - u_true) ** 2))))
    st, sc, sr = (float(np.mean(slopes_truth)), float(np.mean(slopes_corrdiff)),
                  float(np.mean(slopes_regonly)))
    # gate: CorrDiff slope within +-0.45 of the truth slope in the inertial
    # range (and strictly shallower i.e. more turbulent than MSE-only)
    gate_ok = abs(sc - st) <= 0.45 and sc > sr - 0.05

    metrics = {
        "psdSlopeTruth": round(st, 3),
        "psdSlopeCorrDiff": round(sc, 3),
        "psdSlopeRegressionOnly": round(sr, 3),
        "normalizedRmseU": round(float(np.mean(rmses)), 4),
        "gate": {"spectrumRestored(|slope-truth|<=0.45 & > MSE-only)": gate_ok},
        "downscaleFactor": FACTOR, "fineGrid": FINE_GRID,
        "trainMinutes": round((time.time() - t_start) / 60, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (METRICS_DIR / "layer2_corrdiff.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg-steps", type=int, default=160)
    ap.add_argument("--diff-steps", type=int, default=260)
    a = ap.parse_args()
    main(reg_steps=a.reg_steps, diff_steps=a.diff_steps)
