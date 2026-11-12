"""Global configuration for TerraGuard-v2."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("TERRAGUARD_ARTIFACTS", REPO_ROOT / "artifacts"))
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
METRICS_DIR = ARTIFACT_DIR / "metrics"
FIGURE_DIR = ARTIFACT_DIR / "figures"
DATA_DIR = Path(os.environ.get("TERRAGUARD_DATA", REPO_ROOT / "data"))

for _d in (CHECKPOINT_DIR, METRICS_DIR, FIGURE_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 1337

# ----- Geography (Karnataka wind-solar belt, per blueprint §4.2) -----
DEFAULT_ANCHOR_LATLON = (14.2250, 76.3980)
REGION_LAT_SLICE = (18.5, 11.5)   # north -> south
REGION_LON_SLICE = (74.0, 78.5)

# ----- Layer 2 grids -----
COARSE_RES_KM = 25.0      # ERA5/GraphCast 0.25 deg
FINE_RES_KM = 2.0         # CorrDiff target
COARSE_GRID = 16          # regional patch (16 x 25 km = 400 km)
FINE_GRID = 128           # 16 * 8 upsample -> 2 km-ish cells (downscale factor 8)
ATMOS_CHANNELS = 6        # u10, v10, t2m, sp, tcc, q

# ----- Layer 1 SDA -----
SDA_GRID = 32
SDA_DIFFUSION_STEPS = 64

# ----- Turbine / solar engineering constants (blueprint §3.1) -----
GCR = 0.45                # ground coverage ratio
HA_PER_MWP = 1.6          # single-axis tracker land packing factor
ROTOR_DIAMETER_M = 127.0  # 120 m hub-height class turbine
TURBINE_MW = 3.3
HUB_HEIGHT_M = 120.0
WAKE_SPACING_ROTORS = 7.0

# ----- Financial defaults (Section D, INR crore basis) -----
SOLAR_CAPEX_CR_PER_MW = 4.0
WIND_CAPEX_CR_PER_MW = 6.5
TX_LINE_CR_PER_KM = 0.35
OM_PCT_OF_CAPEX = 0.015
DISCOUNT_RATE = 0.082
PPA_TARIFF_INR_KWH = 3.10
PROJECT_LIFE_YEARS = 25
DEGRADATION_PCT_YR = 0.006

DEVICE = os.environ.get("TERRAGUARD_DEVICE", "cpu")
