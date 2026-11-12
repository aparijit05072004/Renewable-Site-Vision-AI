"""Geolocation & spatial-scale resolver (blueprint §3.1 step 1).

Priority chain:
  1. GeoTIFF (.tif/.tiff)  -> embedded CRS + affine transform via rasterio
  2. JPEG with EXIF GPSInfo -> decimal degrees + altitude-derived GSD
  3. Plain screenshot       -> anchor to the user's selected map pin
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict

from PIL import Image, ExifTags

try:  # rasterio is optional; GeoTIFF support degrades gracefully without it
    import rasterio  # type: ignore
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False

_GPS_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), 34853)


@dataclass
class GeoAnchor:
    lat: float
    lon: float
    gsd_m: float            # ground sampling distance, metres / pixel
    source: str             # geotiff | exif | map_pin
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


def _rational(v) -> float:
    try:
        return float(v)
    except TypeError:
        return float(v[0]) / float(v[1])


def _dms_to_deg(dms, ref: str) -> float:
    deg = _rational(dms[0]) + _rational(dms[1]) / 60.0 + _rational(dms[2]) / 3600.0
    return -deg if ref in ("S", "W") else deg


def resolve_image_geolocation(file_bytes: bytes, file_name: str,
                              fallback_latlon: tuple[float, float] = (14.2250, 76.3980),
                              fallback_gsd_m: float = 10.0) -> GeoAnchor:
    name = file_name.lower()
    if name.endswith((".tif", ".tiff")) and _HAS_RASTERIO:
        with rasterio.open(io.BytesIO(file_bytes)) as ds:
            b = ds.bounds
            lat = 0.5 * (b.bottom + b.top)
            lon = 0.5 * (b.left + b.right)
            gsd = abs(ds.res[0])
            if ds.crs and ds.crs.is_geographic:   # degrees -> metres
                gsd = gsd * 111_320.0
            return GeoAnchor(lat, lon, float(gsd), "geotiff", ds.width, ds.height)

    img = Image.open(io.BytesIO(file_bytes))
    w, h = img.size
    try:
        exif = img.getexif()
        gps = exif.get_ifd(_GPS_TAG) if exif else {}
        if gps and 2 in gps and 4 in gps:
            lat = _dms_to_deg(gps[2], str(gps.get(1, "N")))
            lon = _dms_to_deg(gps[4], str(gps.get(3, "E")))
            # DJI-style relative altitude -> GSD estimate (24 mm eq., 4:3 sensor)
            alt = _rational(gps[6]) if 6 in gps else 120.0
            gsd = max(0.02, (alt * 0.0088) / 0.0088 * 0.000173 * 100)  # ~1.7 cm @ 100 m
            return GeoAnchor(lat, lon, gsd, "exif", w, h)
    except Exception:
        pass
    return GeoAnchor(fallback_latlon[0], fallback_latlon[1], fallback_gsd_m,
                     "map_pin", w, h)
