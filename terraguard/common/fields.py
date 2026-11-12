"""Procedural physics-consistent field generators.

These generators stand in for the open-data feeds named in the blueprint
(ARCO-ERA5, NIWE 120 m Wind Atlas, NREL NSRDB, Copernicus DEM GLO-30,
ESA WorldCover) so that every TerraGuard-v2 model can be trained and
validated end-to-end on CPU without multi-terabyte downloads.  The
spectral statistics are matched to the real phenomena: atmospheric
mesoscale fields follow a Kolmogorov -5/3 kinetic-energy cascade, and
terrain follows a red (k^-2) spectrum, which is what the CorrDiff
spectral gate (blueprint Table 4, Phase 3) verifies.

The real-data extraction scripts live in ``terraguard/data_acquisition``
and produce tensors of the same shapes, so swapping in actual ERA5/NSRDB
archives is a drop-in change.
"""
from __future__ import annotations

import numpy as np


def _radial_wavenumber(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(k, k)
    kr = np.sqrt(kx**2 + ky**2)
    kr[0, 0] = 1e-6
    return kr


def spectral_field(n: int, slope: float, rng: np.random.Generator,
                   anisotropy: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """Gaussian random field with isotropic power spectrum E(k) ~ k^slope.

    For a 2-D field whose kinetic-energy spectrum should follow k^-5/3,
    the amplitude spectrum must scale as k^-(5/3+1)/2 per dimension; we
    parameterise directly on the *energy* slope for clarity.
    """
    kr = _radial_wavenumber(n)
    if anisotropy > 0.0:
        k = np.fft.fftfreq(n) * n
        kx, ky = np.meshgrid(k, k)
        kxr = kx * np.cos(angle) + ky * np.sin(angle)
        kyr = -kx * np.sin(angle) + ky * np.cos(angle)
        kr = np.sqrt((kxr * (1 + anisotropy))**2 + kyr**2)
        kr[0, 0] = 1e-6
    # energy E(k) ~ k^slope  ->  |F|^2 * k ~ k^slope  ->  |F| ~ k^((slope-1)/2)
    amp = kr ** ((slope - 1.0) / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=(n, n))
    f = amp * np.exp(1j * phase)
    field = np.fft.ifft2(f).real
    field -= field.mean()
    std = field.std()
    return field / (std + 1e-12)


def radially_averaged_psd(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, E(k)) — the radially averaged kinetic-energy spectrum."""
    n = field.shape[-1]
    f = np.fft.fft2(field)
    psd2 = (np.abs(f) ** 2) / (n * n)
    kr = _radial_wavenumber(n)
    kbins = np.arange(1, n // 2)
    e = np.zeros(len(kbins))
    for i, kb in enumerate(kbins):
        mask = (kr >= kb - 0.5) & (kr < kb + 0.5)
        # ring-integrated energy  E(k) = sum |F|^2 over the ring
        e[i] = psd2[mask].sum()
    return kbins.astype(float), e


def psd_slope(field: np.ndarray, kmin: int = 4, kmax: int | None = None) -> float:
    """Least-squares log-log slope of the energy spectrum over the inertial range."""
    k, e = radially_averaged_psd(field)
    if kmax is None:
        kmax = field.shape[-1] // 3
    sel = (k >= kmin) & (k <= kmax) & (e > 0)
    return float(np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)[0])


def synthetic_dem(n: int, rng: np.random.Generator, relief_m: float = 420.0,
                  base_elev_m: float = 600.0) -> np.ndarray:
    """30 m-class DEM patch with ridgelines (red spectrum + ridged noise)."""
    base = spectral_field(n, -2.0, rng)
    ridged = 1.0 - np.abs(spectral_field(n, -1.7, rng, anisotropy=1.4,
                                         angle=rng.uniform(0, np.pi)))
    dem = 0.65 * base + 0.35 * (ridged - ridged.mean()) / (ridged.std() + 1e-9)
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-12)
    return base_elev_m + dem * relief_m


def terrain_slope_deg(dem: np.ndarray, cell_m: float = 30.0) -> np.ndarray:
    gy, gx = np.gradient(dem, cell_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def coarse_atmos_state(n: int, rng: np.random.Generator,
                       channels: int = 6) -> np.ndarray:
    """Coarse 0.25-deg style atmospheric state [C,H,W].

    Channels: u10 (m/s), v10, t2m (degC), sp (hPa anomaly), tcc [0,1], q (g/kg).
    Large-scale smooth (synoptic k^-3 enstrophy range) fields.
    """
    monsoonal_u = rng.uniform(2.0, 9.0)
    monsoonal_v = rng.uniform(-3.0, 3.0)
    out = np.zeros((channels, n, n), dtype=np.float32)
    out[0] = monsoonal_u + 2.2 * spectral_field(n, -3.0, rng)
    out[1] = monsoonal_v + 2.2 * spectral_field(n, -3.0, rng)
    out[2] = rng.uniform(22.0, 34.0) + 1.8 * spectral_field(n, -3.0, rng)
    out[3] = 4.0 * spectral_field(n, -3.0, rng)
    cc = 0.35 + 0.28 * spectral_field(n, -2.6, rng)
    out[4] = np.clip(cc, 0.0, 1.0)
    out[5] = rng.uniform(8.0, 16.0) + 1.1 * spectral_field(n, -2.8, rng)
    return out


def fine_from_coarse(coarse: np.ndarray, factor: int, dem_norm: np.ndarray,
                     rng: np.random.Generator,
                     turb_amp: float = 1.15) -> np.ndarray:
    """Build the 'true' 2 km field a CorrDiff model must learn.

    truth = bilinear(coarse) + terrain forcing + k^-5/3 stochastic turbulence.
    """
    c, h, w = coarse.shape
    n = h * factor
    up = np.zeros((c, n, n), dtype=np.float32)
    for ci in range(c):
        up[ci] = _bilinear_upsample(coarse[ci], factor)
    turb_u = spectral_field(n, -5.0 / 3.0, rng)
    turb_v = spectral_field(n, -5.0 / 3.0, rng)
    gy, gx = np.gradient(dem_norm)
    speedup = 1.0 + 1.8 * np.clip(-gx, 0, None)  # upslope acceleration proxy
    fine = up.copy()
    fine[0] = up[0] * speedup + turb_amp * turb_u
    fine[1] = up[1] * speedup + turb_amp * turb_v
    fine[2] = up[2] - 6.5 * dem_norm * 0.42  # lapse-rate cooling with elevation
    fine[3] = up[3] - 11.0 * dem_norm       # hydrostatic pressure decrease
    fine[4] = np.clip(up[4] + 0.15 * spectral_field(n, -5.0 / 3.0, rng)
                      + 0.10 * dem_norm, 0, 1)  # orographic cloud
    fine[5] = up[5] + 0.6 * spectral_field(n, -5.0 / 3.0, rng)
    return fine


def _bilinear_upsample(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape
    xs = (np.arange(h * f) + 0.5) / f - 0.5
    x0 = np.clip(np.floor(xs).astype(int), 0, h - 2)
    t = xs - x0
    rows = a[x0] * (1 - t)[:, None] + a[x0 + 1] * t[:, None]
    cols = rows[:, x0] * (1 - t)[None, :] + rows[:, x0 + 1] * t[None, :]
    return cols.astype(np.float32)


def weibull_params_from_mean(v_mean: float, k_shape: float = 2.1) -> tuple[float, float]:
    """Return Weibull (A, k) whose mean matches v_mean."""
    from math import gamma
    a = v_mean / gamma(1.0 + 1.0 / k_shape)
    return a, k_shape
