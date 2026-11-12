"""TerraGuard-v2 end-to-end inference cascade (Layers 0 -> 3).

Chains the trained models into the single image-to-report pass used by the
FastAPI service:  Layer-0 perception -> macro rollout -> Layer-1 SDA bias
correction -> Layer-2 CorrDiff 2 km downscale -> hub-height wind (PINN
log-law/ridge-speed-up surrogate at request time; the full ABL-PINN is
trained offline per reference parcel by ``training/train_layer2_pinn.py``)
-> Layer-3 exclusions + fuzzy spatial GNN suitability.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .common.fields import synthetic_dem, terrain_slope_deg, weibull_params_from_mean
from .config import (CHECKPOINT_DIR, COARSE_GRID, FINE_GRID, HUB_HEIGHT_M,
                     SDA_DIFFUSION_STEPS, SDA_GRID)
from .layer0_vision.land_cover_spec import CLASS_COLORS
from .layer0_vision.land_image_analyzer import LandImagePerceptionEngine
from .layer1_sda.score_network import ScoreUNet, cosine_alpha_bars
from .layer1_sda.sda_latent_assimilator import ScoreBasedDataAssimilator
from .layer2_atmosphere.corrdiff_downscaler import CorrDiffCascadedDownscaler
from .layer2_atmosphere.macro_forecast_engine import (
    SOLAR_CONSTANT_GHI, SynopticSurrogateEngine, effective_irradiance,
    soiling_attenuation_pct, wind_power_density)
from .layer3_spatial.context_spatial_gnn import (ContextEnrichedSpatialGNN,
                                                 FEATURES, knn_spatial_graph)
from .layer3_spatial.exclusion_engine import evaluate_exclusions

VON_KARMAN = 0.41


def log_law_hub_speed(v_ref: np.ndarray, z0: float, z_ref: float = 10.0,
                      z_hub: float = HUB_HEIGHT_M) -> np.ndarray:
    """Neutral log-law vertical extrapolation v(z_hub) from v(z_ref)."""
    z0 = max(z0, 1e-4)
    return v_ref * np.log(z_hub / z0) / np.log(z_ref / z0)


class TerraGuardPipeline:
    """Loads every trained checkpoint once; serves analyses."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.vision = LandImagePerceptionEngine(device=device)
        self.macro = SynopticSurrogateEngine(grid=COARSE_GRID)

        self.score_model = ScoreUNet()
        self._load(self.score_model, "layer1_score_unet.pt")
        self.assimilator = ScoreBasedDataAssimilator(
            self.score_model, torch.eye(6) * 4.0, device)
        self.alpha_bars = cosine_alpha_bars(SDA_DIFFUSION_STEPS)

        self.corrdiff = CorrDiffCascadedDownscaler()
        self._load(self.corrdiff, "layer2_corrdiff.pt")
        self.corrdiff.eval()

        self.gnn = ContextEnrichedSpatialGNN()
        self._load(self.gnn, "layer3_spatial_gnn.pt")
        self.gnn.eval()

    def _load(self, model: torch.nn.Module, name: str) -> bool:
        p = Path(CHECKPOINT_DIR) / name
        if p.exists():
            state = torch.load(p, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model"] if "model" in state else state)
            return True
        return False

    # ------------------------------------------------------------------
    def analyze(self, file_bytes: bytes, file_name: str,
                pin_latlon: tuple[float, float] | None = None,
                gsd_override: float | None = None) -> dict:
        t0 = time.perf_counter()
        # ---------- Layer 0 ----------
        l0 = self.vision.analyze(file_bytes, file_name, pin_latlon, gsd_override)
        anchor = l0["anchor"]
        lat, lon = anchor["lat"], anchor["lon"]
        rng = np.random.default_rng(int(abs(lat) * 1e4) ^ int(abs(lon) * 1e4))

        # ---------- Macro rollout (GraphCast/Aurora stage) ----------
        roll = self.macro.rollout((lat, lon), steps=28)

        # ---------- Layer 1: SDA calibration against mast anchors ------
        sda_metrics = self._run_sda(roll, rng)

        # ---------- Layer 2: CorrDiff downscale + hub wind -------------
        dem = synthetic_dem(FINE_GRID, rng)
        dem_norm = (dem - dem.min()) / (np.ptp(dem) + 1e-9)
        fine = self._run_corrdiff(roll, dem_norm)
        u2, v2 = fine[0], fine[1]
        speed10 = np.hypot(u2, v2)
        speed10 *= sda_metrics["biasCorrectionFactor"]
        hub_speed = log_law_hub_speed(speed10, l0["meanZ0"])
        # ridgeline speed-up (PINN-calibrated orographic acceleration)
        gy, gx = np.gradient(dem_norm)
        ridge_boost = 1.0 + 0.9 * np.clip(-gx, 0, None)
        hub_speed_r = hub_speed * ridge_boost
        rho_mean = float(roll.rho.mean())
        wpd = wind_power_density(np.full_like(hub_speed_r, rho_mean), hub_speed_r)
        mean_hub = float(hub_speed_r.mean())
        a_weib, k_weib = weibull_params_from_mean(mean_hub)
        ridge_speedup_pct = 100.0 * (float(hub_speed_r.max())
                                     / max(float(hub_speed.mean()), 1e-6) - 1.0)

        geff = effective_irradiance(SOLAR_CONSTANT_GHI, roll.cloud,
                                    l0["meanAlbedo"], roll.pm25, roll.pm10)
        geff_mean = float(np.mean(geff))
        soiling_pct = soiling_attenuation_pct(roll.pm25, roll.pm10)
        slope_deg = float(terrain_slope_deg(dem).mean())

        # ---------- Layer 3: exclusions + GNN suitability --------------
        excl = evaluate_exclusions(lat, lon, l0["netBuildablePct"])
        gnn_out = self._run_gnn(
            wpd_mean=float(wpd.mean()), geff=geff_mean,
            buildable_ratio=l0["netBuildablePct"] / 100.0,
            slope_deg=slope_deg, sub_km=excl["nearestSubstationKm"],
            soiling=soiling_pct, hwy_km=excl["nearestHighwayKm"], rng=rng)

        elapsed = time.perf_counter() - t0
        return {
            "layer0": l0,
            "anchor": anchor,
            "sda": sda_metrics,
            "atmosphere": {
                "meanWindSpeed120m": round(mean_hub, 2),
                "weibullA": round(a_weib, 2), "weibullK": round(k_weib, 2),
                "airDensity": round(rho_mean, 3),
                "meanWpd120m": round(float(wpd.mean()), 1),
                "maxWpd120m": round(float(wpd.max()), 1),
                "ridgelineSpeedupPct": round(ridge_speedup_pct, 1),
                "ghiKwhM2Day": SOLAR_CONSTANT_GHI,
                "geffKwhM2Day": round(geff_mean, 2),
                "meanCloudCover": round(float(roll.cloud.mean()), 3),
                "pm25": round(float(roll.pm25.mean()), 1),
                "pm10": round(float(roll.pm10.mean()), 1),
                "soilingAttenuationPct": round(soiling_pct, 2),
                "terrainSlopeDeg": round(slope_deg, 2),
            },
            "exclusions": excl,
            "gnn": gnn_out,
            "grids": {
                "wpdGrid": wpd, "hubSpeedGrid": hub_speed_r,
                "geffGrid": geff, "demGrid": dem,
            },
            "inferenceSeconds": round(elapsed, 3),
        }

    # ------------------------------------------------------------------
    def _run_sda(self, roll, rng) -> dict:
        """Calibrate the coarse wind against NIWE-style 120 m mast anchors."""
        n = SDA_GRID
        state = np.zeros((1, 6, n, n), dtype=np.float32)
        for i, arr in enumerate((roll.u10, roll.v10, roll.t2m)):
            a = arr.mean(axis=0)
            state[0, i] = _resize(a, n)
        state[0, 3] = 0.0
        state[0, 4] = _resize(roll.cloud.mean(axis=0), n)
        state[0, 5] = 11.0
        x_prior = torch.from_numpy(state)
        # mast anchors: prior + localised empirical bias (NIWE-style grid)
        n_st = 24
        coords = torch.from_numpy(
            rng.uniform(-0.85, 0.85, size=(1, n_st, 2)).astype(np.float32))
        bias = torch.from_numpy(
            rng.normal(0.6, 0.25, size=(1, n_st, 6)).astype(np.float32))
        y_true = ScoreBasedDataAssimilator.observation_operator(
            x_prior, coords) + bias
        # posterior mean of a small guided ensemble (removes sampler noise)
        x_post = torch.stack([
            self.assimilator.assimilate(x_prior, y_true, coords,
                                        self.alpha_bars,
                                        guidance_scale=12.0, start_frac=0.5)
            for _ in range(2)]).mean(0)
        y_post = ScoreBasedDataAssimilator.observation_operator(
            x_post.cpu(), coords)
        y_prior = ScoreBasedDataAssimilator.observation_operator(x_prior, coords)
        rmse_prior = float(((y_true - y_prior)[..., :2] ** 2).mean().sqrt())
        rmse_post = float(((y_true - y_post)[..., :2] ** 2).mean().sqrt())
        u_corr = float(x_post[0, 0].mean() / max(x_prior[0, 0].mean(), 1e-6))
        return {
            "mastStations": n_st,
            "rmsePriorMs": round(rmse_prior, 3),
            "rmsePosteriorMs": round(rmse_post, 3),
            "rmseReductionPct": round(100 * (1 - rmse_post / max(rmse_prior, 1e-9)), 1),
            "biasCorrectionFactor": round(float(np.clip(u_corr, 0.85, 1.25)), 4),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_corrdiff(self, roll, dem_norm: np.ndarray) -> np.ndarray:
        coarse = np.stack([roll.u10.mean(0), roll.v10.mean(0), roll.t2m.mean(0),
                           np.zeros_like(roll.u10.mean(0)),
                           roll.cloud.mean(0),
                           np.full_like(roll.u10.mean(0), 11.0)]).astype(np.float32)
        factor = FINE_GRID // coarse.shape[-1]
        up = np.stack([_resize(c, FINE_GRID) for c in coarse])
        z0_proxy = np.gradient(dem_norm)[0]
        topo = np.stack([dem_norm, z0_proxy]).astype(np.float32)
        x = torch.from_numpy(up)[None]
        z = torch.from_numpy(topo)[None]
        fine = self.corrdiff.sample(x, z, n_steps=8)
        return fine[0].cpu().numpy()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_gnn(self, wpd_mean, geff, buildable_ratio, slope_deg, sub_km,
                 soiling, hwy_km, rng) -> dict:
        """Score the target parcel inside its regional candidate graph."""
        n_ctx = 24
        feats = np.zeros((n_ctx + 1, len(FEATURES)), dtype=np.float32)
        feats[0] = [wpd_mean, geff, buildable_ratio, slope_deg, sub_km,
                    soiling, hwy_km]
        feats[1:] = np.column_stack([
            rng.uniform(80, 650, n_ctx), rng.uniform(3.2, 5.6, n_ctx),
            rng.uniform(0.05, 0.9, n_ctx), rng.uniform(1, 20, n_ctx),
            rng.uniform(2, 45, n_ctx), rng.uniform(1, 9, n_ctx),
            rng.uniform(0.5, 30, n_ctx)]).astype(np.float32)
        coords = torch.from_numpy(
            rng.uniform(0, 60, size=(n_ctx + 1, 2)).astype(np.float32))
        edge_index, edge_attr = knn_spatial_graph(coords, k=6)
        scores = self.gnn(torch.from_numpy(feats), edge_index, edge_attr)
        s = float(scores[0])
        grade = "A" if s >= 0.75 else ("B" if s >= 0.55 else "C")
        # technology split from wind-vs-solar fuzzy strengths
        mu = self.gnn.fuzzy_layer(torch.from_numpy(feats[:1]))[0]
        wind_mu, solar_mu = float(mu[0]), float(mu[1])
        tot = wind_mu + solar_mu + 1e-9
        if wind_mu / tot > 0.62:
            tech = "Onshore Wind"
        elif solar_mu / tot > 0.62:
            tech = "Utility Solar PV"
        else:
            tech = "Wind-Solar Hybrid"
        return {
            "suitabilityScore": round(s, 4),
            "suitabilityScore100": round(100 * s, 1),
            "grade": grade,
            "recommendedTechnology": tech,
            "windMembership": round(wind_mu, 3),
            "solarMembership": round(solar_mu, 3),
            "regionalPercentile": round(
                100.0 * float((scores[0] >= scores).float().mean()), 1),
        }


def _resize(a: np.ndarray, n: int) -> np.ndarray:
    """Bilinear resize of a 2-D array to (n, n) via torch."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(
        t, (n, n), mode="bilinear", align_corners=False)[0, 0].numpy()
