"""Layer 0 Perception Engine (blueprint §3.2).

Resolves coordinates from uploaded land imagery, segments pixels into the
7 siting classes, applies Euclidean-dilation regulatory setback buffers,
extracts z0(x,y) and albedo grids for the Layer-2 PINN, and computes net
buildable area / installable capacity.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from ..config import (CHECKPOINT_DIR, GCR, HA_PER_MWP, ROTOR_DIAMETER_M,
                      TURBINE_MW, WAKE_SPACING_ROTORS)
from .geo_metadata_resolver import GeoAnchor, resolve_image_geolocation
from .land_cover_spec import (ALBEDO_LUT, BUFFER_LUT, CLASS_NAMES,
                              LAND_COVER_SPEC, NUM_CLASSES, Z0_LUT)
from .segmentation_model import SegFormerNano

_INPUT_SIZE = 256


class LandImagePerceptionEngine:
    """End-to-end Layer-0 analysis for a single uploaded land image."""

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.device = device
        self.model = SegFormerNano(num_classes=NUM_CLASSES).to(device)
        ckpt = Path(checkpoint) if checkpoint else CHECKPOINT_DIR / "layer0_segformer_nano.pt"
        self.checkpoint_loaded = False
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            self.model.load_state_dict(state["model"])
            self.checkpoint_loaded = True
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """rgb uint8 [H,W,3] -> class mask int64 [H,W] (at native size)."""
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb).resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None].to(self.device)
        logits = self.model(x)
        mask = logits.argmax(1)[0].cpu().numpy().astype(np.int64)
        return np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST),
            dtype=np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def buildable_mask(seg: np.ndarray, gsd_m: float) -> np.ndarray:
        """Boolean mask after Euclidean setback dilation around restricted classes."""
        build = np.isin(seg, [c for c, s in LAND_COVER_SPEC.items() if s["buildable"]])
        for cls, buf in enumerate(BUFFER_LUT):
            if buf <= 0:
                continue
            obstacle = seg == cls
            if not obstacle.any():
                continue
            dist_px = distance_transform_edt(~obstacle)
            build &= dist_px * gsd_m >= buf
        return build

    # ------------------------------------------------------------------
    def analyze(self, file_bytes: bytes, file_name: str,
                pin_latlon: tuple[float, float] | None = None,
                gsd_override: float | None = None) -> dict:
        anchor: GeoAnchor = resolve_image_geolocation(
            file_bytes, file_name,
            fallback_latlon=pin_latlon or (14.2250, 76.3980))
        if gsd_override:
            anchor.gsd_m = gsd_override
        rgb = np.asarray(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
        seg = self.segment(rgb)
        build = self.buildable_mask(seg, anchor.gsd_m)

        z0 = np.asarray(Z0_LUT, dtype=np.float32)[seg]
        albedo = np.asarray(ALBEDO_LUT, dtype=np.float32)[seg]

        px_area_ha = anchor.gsd_m ** 2 / 10_000.0
        total_ha = seg.size * px_area_ha
        buildable_ha = float(build.sum() * px_area_ha)
        buildable_pct = 100.0 * build.mean()

        class_pct = {CLASS_NAMES[c]: round(100.0 * float((seg == c).mean()), 2)
                     for c in range(NUM_CLASSES) if (seg == c).any()}

        p_solar_mwp = buildable_ha * GCR / HA_PER_MWP
        # ridgeline proxy: longest buildable row span (metres) for turbine string
        row_lengths = build.sum(axis=1) * anchor.gsd_m
        l_ridge_m = float(row_lengths.max()) if row_lengths.size else 0.0
        n_turbines = int(l_ridge_m // (WAKE_SPACING_ROTORS * ROTOR_DIAMETER_M))
        p_wind_mw = n_turbines * TURBINE_MW

        return {
            "anchor": anchor.to_dict(),
            "segmentation": seg,
            "buildableMask": build,
            "z0Grid": z0,
            "albedoGrid": albedo,
            "meanZ0": float(z0.mean()),
            "meanAlbedo": float(albedo.mean()),
            "totalFootprintHa": round(total_ha, 2),
            "netBuildableHa": round(buildable_ha, 2),
            "netBuildablePct": round(buildable_pct, 2),
            "classDistributionPct": class_pct,
            "pSolarMwp": round(p_solar_mwp, 2),
            "nTurbines": n_turbines,
            "pWindMw": round(p_wind_mw, 2),
            "roadAccess": bool((seg == 6).any()),
            "setbacksAppliedM": {CLASS_NAMES[c]: BUFFER_LUT[c]
                                 for c in range(NUM_CLASSES) if BUFFER_LUT[c] > 0},
            "checkpointLoaded": self.checkpoint_loaded,
        }
