"""Automated solar/wind layout blueprint renderer (blueprint §3.2, Section A).

Draws south-facing single-axis solar PV tracker rows over flat buildable
pixels and 120 m wind-turbine micro-siting markers (7D downwind x 4D
crosswind wake spacing) along the high-wind ridgeline, directly on the
user's uploaded land image.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from ..config import ROTOR_DIAMETER_M, WAKE_SPACING_ROTORS

PV_COLOR = (37, 99, 235, 215)
PV_EDGE = (191, 219, 254, 255)
TURBINE_COLOR = (250, 250, 250, 255)
TURBINE_RING = (16, 185, 129, 220)


def render_layout_overlay(rgb: np.ndarray, buildable: np.ndarray,
                          gsd_m: float, n_turbines: int,
                          wind_dir_deg: float = 250.0) -> bytes:
    """Return PNG bytes of the annotated layout blueprint."""
    h, w = buildable.shape
    base = Image.fromarray(rgb).convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    # --- Solar PV tracker rows (east-west rows, south-facing tilt) ---
    row_pitch_px = max(3, int(round(6.0 / max(gsd_m, 0.5))))   # ~6 m pitch
    row_h = max(1, row_pitch_px // 3)
    for y in range(0, h - row_h, row_pitch_px):
        spans = _row_spans(buildable[y:y + row_h].all(axis=0))
        for x0, x1 in spans:
            if (x1 - x0) * gsd_m < 12.0:     # skip slivers < 12 m
                continue
            draw.rectangle([x0, y, x1, y + row_h], fill=PV_COLOR,
                           outline=PV_EDGE, width=1)

    # --- Wind turbines along longest buildable ridgeline row ---
    spacing_px = max(4, int(WAKE_SPACING_ROTORS * ROTOR_DIAMETER_M / max(gsd_m, 0.5)))
    row_scores = buildable.sum(axis=1)
    if n_turbines > 0 and row_scores.max() > 0:
        y_t = int(row_scores.argmax())
        xs = np.flatnonzero(buildable[y_t])
        placed = 0
        x = xs[0] if xs.size else 0
        r = max(3, spacing_px // 8)
        while placed < n_turbines and x <= (xs[-1] if xs.size else 0):
            if buildable[y_t, min(x, w - 1)]:
                _draw_turbine(draw, x, y_t, r)
                placed += 1
                x += spacing_px
            else:
                x += 2
    # wind direction arrow
    _draw_wind_arrow(draw, w, h, wind_dir_deg)

    out = Image.alpha_composite(base, ov).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def render_segmentation_overlay(rgb: np.ndarray, seg: np.ndarray,
                                colors: list[tuple[int, int, int]],
                                alpha: float = 0.45) -> bytes:
    """Return PNG bytes of the 7-class semantic overlay."""
    palette = np.asarray(colors, dtype=np.float32)
    tinted = (1 - alpha) * rgb.astype(np.float32) + alpha * palette[seg]
    buf = io.BytesIO()
    Image.fromarray(np.clip(tinted, 0, 255).astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _row_spans(row: np.ndarray) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, v in enumerate(row):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(row) - 1))
    return spans


def _draw_turbine(draw: ImageDraw.ImageDraw, x: int, y: int, r: int) -> None:
    draw.ellipse([x - r, y - r, x + r, y + r], outline=TURBINE_RING, width=2)
    for ang in (90.0, 210.0, 330.0):
        a = np.radians(ang)
        draw.line([x, y, x + r * 0.9 * np.cos(a), y - r * 0.9 * np.sin(a)],
                  fill=TURBINE_COLOR, width=2)
    draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=TURBINE_COLOR)


def _draw_wind_arrow(draw: ImageDraw.ImageDraw, w: int, h: int, deg: float) -> None:
    cx, cy, r = w - 36, 36, 20
    a = np.radians(270.0 - deg)
    x1, y1 = cx - r * np.cos(a), cy + r * np.sin(a)
    x2, y2 = cx + r * np.cos(a), cy - r * np.sin(a)
    draw.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4],
                 outline=(255, 255, 255, 200), width=2)
    draw.line([x1, y1, x2, y2], fill=(245, 158, 11, 255), width=3)
    head = 6
    draw.polygon([(x2, y2),
                  (x2 - head * np.cos(a - 0.5), y2 + head * np.sin(a - 0.5)),
                  (x2 - head * np.cos(a + 0.5), y2 + head * np.sin(a + 0.5))],
                 fill=(245, 158, 11, 255))
