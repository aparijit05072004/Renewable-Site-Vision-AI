"""Phase 2 — Train the SDA score network & verify the assimilation gate.

Gate (blueprint Table 4): >= 25% RMSE reduction on held-out mast wind
speeds versus raw bilinear ERA5 interpolation.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..common.fields import coarse_atmos_state
from ..config import CHECKPOINT_DIR, METRICS_DIR, SDA_DIFFUSION_STEPS, SDA_GRID, SEED
from ..layer1_sda.score_network import ScoreUNet, cosine_alpha_bars
from ..layer1_sda.sda_latent_assimilator import ScoreBasedDataAssimilator

CHANNEL_SCALE = np.array([3.0, 3.0, 3.0, 4.0, 0.3, 2.0], dtype=np.float32)
CHANNEL_MEAN = np.array([5.5, 0.0, 28.0, 0.0, 0.4, 12.0], dtype=np.float32)


def sample_states(batch: int, rng: np.random.Generator) -> torch.Tensor:
    out = np.zeros((batch, 6, SDA_GRID, SDA_GRID), dtype=np.float32)
    for b in range(batch):
        out[b] = coarse_atmos_state(SDA_GRID, rng)
    out = (out - CHANNEL_MEAN[None, :, None, None]) / CHANNEL_SCALE[None, :, None, None]
    return torch.from_numpy(out)


def main(steps: int = 500, batch: int = 16, lr: float = 2e-4,
         seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ScoreUNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ab = cosine_alpha_bars(SDA_DIFFUSION_STEPS)

    t_start = time.time()
    model.train()
    for step in range(steps):
        x0 = sample_states(batch, rng)
        t = torch.randint(0, SDA_DIFFUSION_STEPS, (batch,))
        noise = torch.randn_like(x0)
        ab_t = ab[t].view(-1, 1, 1, 1)
        xt = ab_t.sqrt() * x0 + (1 - ab_t).sqrt() * noise
        pred = model(xt, t)
        loss = torch.nn.functional.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"[L1] step {step:4d}/{steps}  eps-mse={loss.item():.4f}")

    torch.save({"model": model.state_dict()},
               CHECKPOINT_DIR / "layer1_score_unet.pt")

    # ------- Assimilation gate: biased prior vs mast truth -------
    model.eval()
    assim = ScoreBasedDataAssimilator(model, torch.eye(6) * 6.0)
    reductions = []
    for trial in range(6):
        truth = sample_states(1, rng)
        # systematic regional bias + smoothing = the "raw ERA5" prior
        bias = torch.from_numpy(
            rng.normal(0.25, 0.1, size=(1, 6, 1, 1)).astype(np.float32))
        prior = torch.nn.functional.avg_pool2d(
            truth, 5, stride=1, padding=2) + bias
        n_st = 8
        coords = torch.from_numpy(
            rng.uniform(-0.9, 0.9, size=(1, n_st, 2)).astype(np.float32))
        y_obs = ScoreBasedDataAssimilator.observation_operator(truth, coords)
        post = assim.assimilate(prior, y_obs, coords, ab,
                                guidance_scale=2.2, start_frac=0.45)
        # held-out masts (different sites than the guided ones)
        coords_h = torch.from_numpy(
            rng.uniform(-0.9, 0.9, size=(1, 12, 2)).astype(np.float32))
        yt = ScoreBasedDataAssimilator.observation_operator(truth, coords_h)
        yp = ScoreBasedDataAssimilator.observation_operator(prior, coords_h)
        ya = ScoreBasedDataAssimilator.observation_operator(post.cpu(), coords_h)
        rmse_prior = float(((yt - yp)[..., :2] ** 2).mean().sqrt())
        rmse_post = float(((yt - ya)[..., :2] ** 2).mean().sqrt())
        reductions.append(100 * (1 - rmse_post / rmse_prior))
        print(f"[L1-gate] trial {trial}: prior {rmse_prior:.4f} -> "
              f"post {rmse_post:.4f}  ({reductions[-1]:+.1f}%)")
    mean_red = float(np.mean(reductions))

    metrics = {
        "rmseReductionPctMean": round(mean_red, 1),
        "rmseReductionPctPerTrial": [round(r, 1) for r in reductions],
        "gate": {"rmseReduction>=25pct": mean_red >= 25.0},
        "diffusionSteps": SDA_DIFFUSION_STEPS,
        "trainSteps": steps,
        "trainMinutes": round((time.time() - t_start) / 60, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (METRICS_DIR / "layer1_sda.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    main(steps=ap.parse_args().steps)
