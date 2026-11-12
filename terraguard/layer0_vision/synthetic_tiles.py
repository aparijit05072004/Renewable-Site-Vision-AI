"""Procedural aerial land-tile generator for Layer-0 segmentation training.

Produces (RGB image, 7-class mask) pairs with the spatial morphology of the
DeepGlobe / LoveDA Rural tiles cited in the blueprint: organic class regions
from thresholded multi-octave noise, meandering water bodies, road polylines,
urban clusters with building speckle, and per-class textures.  These stand in
for the Kaggle/Zenodo downloads so the SegFormer-style network can be
fine-tuned and gated (mIoU >= 0.84 on Barren/Scrub/Water/Forest) on CPU.
"""
from __future__ import annotations

import numpy as np

from ..common.fields import spectral_field
from .land_cover_spec import NUM_CLASSES

# base RGB tone per class (will be texture-modulated)
_TONES = {
    0: (168, 148, 138),   # urban: grey-brown rooftops
    1: (146, 158, 92),    # agri: olive fields
    2: (150, 134, 86),    # scrub: tan-green
    3: (44, 74, 40),      # forest: dark green
    4: (52, 86, 110),     # water: deep blue-grey
    5: (196, 172, 128),   # barren: pale sand
    6: (120, 118, 116),   # road: asphalt grey
}


def _region_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    """Base landscape partition among agri(1)/scrub(2)/forest(3)/barren(5)."""
    f1 = spectral_field(n, -2.6, rng)
    f2 = spectral_field(n, -2.2, rng)
    mask = np.full((n, n), 2, dtype=np.int64)          # scrub default
    mask[f1 > 0.55] = 3                                 # forest patches
    mask[f1 < -0.65] = 5                                # barren patches
    mask[(np.abs(f1) <= 0.55) & (f2 > 0.45)] = 1        # field blocks
    return mask


def _carve_water(mask: np.ndarray, rng: np.random.Generator) -> None:
    n = mask.shape[0]
    if rng.random() < 0.75:
        w = spectral_field(n, -3.0, rng)
        mask[w > np.quantile(w, 0.965)] = 4             # lakes / tanks
    if rng.random() < 0.5:                               # meandering stream
        x = rng.integers(0, n)
        y = 0
        width = rng.integers(1, 3)
        while y < n:
            mask[max(0, y):min(n, y + 2),
                 max(0, x - width):min(n, x + width + 1)] = 4
            x += rng.integers(-2, 3)
            x = int(np.clip(x, 0, n - 1))
            y += 1


def _draw_roads(mask: np.ndarray, rng: np.random.Generator) -> None:
    n = mask.shape[0]
    for _ in range(rng.integers(1, 4)):
        horizontal = rng.random() < 0.5
        pos = rng.integers(n // 8, n - n // 8)
        drift = 0.0
        for t in range(n):
            drift += rng.normal(0, 0.35)
            p = int(np.clip(pos + drift, 1, n - 2))
            if horizontal:
                mask[p - 1:p + 1, t] = 6
            else:
                mask[t, p - 1:p + 1] = 6


def _plant_urban(mask: np.ndarray, rng: np.random.Generator) -> None:
    n = mask.shape[0]
    for _ in range(rng.integers(0, 3)):
        cx, cy = rng.integers(n // 8, n - n // 8, size=2)
        r = rng.integers(n // 16, n // 7)
        yy, xx = np.ogrid[:n, :n]
        blob = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        jitter = spectral_field(n, -1.5, rng) > -0.3
        mask[blob & jitter] = 0


def _texture(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = mask.shape[0]
    img = np.zeros((n, n, 3), dtype=np.float32)
    detail = spectral_field(n, -1.2, rng)
    shading = spectral_field(n, -2.5, rng)
    for c in range(NUM_CLASSES):
        sel = mask == c
        if not sel.any():
            continue
        tone = np.array(_TONES[c], dtype=np.float32)
        img[sel] = tone[None, :]
    img += 14.0 * detail[..., None]
    img += 10.0 * shading[..., None]
    # field-row striping on agriculture
    rows = (np.sin(np.arange(n)[:, None] * 0.9 + np.arange(n)[None, :] * 0.25) > 0)
    img[(mask == 1) & rows] *= 0.92
    # building speckle on urban
    speck = rng.random((n, n)) > 0.6
    img[(mask == 0) & speck] += 36.0
    # water glint
    img[mask == 4] += 8.0 * spectral_field(n, -2.0, rng)[mask == 4, None]
    img += rng.normal(0, 3.5, size=img.shape)           # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_tile(n: int = 256, seed: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 [H,W,3], mask int64 [H,W])."""
    rng = np.random.default_rng(seed)
    mask = _region_mask(n, rng)
    _carve_water(mask, rng)
    _plant_urban(mask, rng)
    _draw_roads(mask, rng)
    rgb = _texture(mask, rng)
    return rgb, mask


def generate_batch(batch: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.zeros((batch, 3, n, n), dtype=np.float32)
    masks = np.zeros((batch, n, n), dtype=np.int64)
    for b in range(batch):
        rgb, m = generate_tile(n, seed * 100003 + b)
        imgs[b] = rgb.transpose(2, 0, 1) / 255.0
        masks[b] = m
    return imgs, masks
