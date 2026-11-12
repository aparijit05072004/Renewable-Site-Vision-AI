"""7-class renewable-siting land-cover specification (blueprint §3.1).

Every pixel class maps deterministically to a buildability flag, a
regulatory setback buffer radius (m), a Davenport–Wieringa aerodynamic
roughness length z0 (m), and a shortwave surface albedo alpha.
"""
from __future__ import annotations

LAND_COVER_SPEC = {
    0: {"name": "Urban / Habitation",      "buildable": False, "z0": 1.200,
        "albedo": 0.18, "bufferM": 150.0, "color": (220, 38, 38)},
    1: {"name": "Agricultural / Fallow",   "buildable": True,  "z0": 0.100,
        "albedo": 0.22, "bufferM": 0.0,   "color": (132, 204, 22)},
    2: {"name": "Rangeland / Scrubland",   "buildable": True,  "z0": 0.050,
        "albedo": 0.25, "bufferM": 0.0,   "color": (16, 185, 129)},
    3: {"name": "Forest / Tree Canopy",    "buildable": False, "z0": 0.850,
        "albedo": 0.14, "bufferM": 100.0, "color": (6, 95, 70)},
    4: {"name": "Water / Wetland",         "buildable": False, "z0": 0.001,
        "albedo": 0.08, "bufferM": 200.0, "color": (2, 132, 199)},
    5: {"name": "Barren / Wasteland",      "buildable": True,  "z0": 0.005,
        "albedo": 0.32, "bufferM": 0.0,   "color": (245, 158, 11)},
    6: {"name": "Roads / Access Track",    "buildable": False, "z0": 0.010,
        "albedo": 0.15, "bufferM": 15.0,  "color": (100, 116, 139)},
}

NUM_CLASSES = len(LAND_COVER_SPEC)
BUILDABLE_CLASSES = tuple(c for c, s in LAND_COVER_SPEC.items() if s["buildable"])
CLASS_NAMES = [LAND_COVER_SPEC[c]["name"] for c in range(NUM_CLASSES)]
CLASS_COLORS = [LAND_COVER_SPEC[c]["color"] for c in range(NUM_CLASSES)]
Z0_LUT = [LAND_COVER_SPEC[c]["z0"] for c in range(NUM_CLASSES)]
ALBEDO_LUT = [LAND_COVER_SPEC[c]["albedo"] for c in range(NUM_CLASSES)]
BUFFER_LUT = [LAND_COVER_SPEC[c]["bufferM"] for c in range(NUM_CLASSES)]
