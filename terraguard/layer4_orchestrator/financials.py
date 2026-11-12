"""Section D — 25-year financial viability & energy-yield projections."""
from __future__ import annotations

import numpy as np

from ..config import (DEGRADATION_PCT_YR, DISCOUNT_RATE, OM_PCT_OF_CAPEX,
                      PPA_TARIFF_INR_KWH, PROJECT_LIFE_YEARS,
                      SOLAR_CAPEX_CR_PER_MW, TURBINE_MW, TX_LINE_CR_PER_KM,
                      WIND_CAPEX_CR_PER_MW)


def solar_cuf(geff_kwh_m2_day: float) -> float:
    """Capacity-utilisation factor from effective daily irradiance
    (PR ~ 0.80 single-axis tracker)."""
    return float(np.clip(geff_kwh_m2_day * 0.80 / 24.0 * 1.25, 0.10, 0.33))


def wind_cuf(mean_speed_120m: float) -> float:
    """CUF from mean hub-height speed via simplified power-curve integral."""
    return float(np.clip(0.087 * mean_speed_120m - 0.12, 0.04, 0.55))


def irr(cashflows: list[float], lo: float = -0.5, hi: float = 1.5) -> float:
    """Bisection IRR of annual cashflows (cf[0] = -CAPEX)."""
    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def project_financials(p_solar_mwp: float, p_wind_mw: float,
                       geff_kwh_m2_day: float, mean_speed_120m: float,
                       tx_line_km: float) -> dict:
    cuf_s = solar_cuf(geff_kwh_m2_day)
    cuf_w = wind_cuf(mean_speed_120m)
    cap_mw = p_solar_mwp + p_wind_mw
    e_solar_gwh = p_solar_mwp * cuf_s * 8760.0 / 1000.0
    e_wind_gwh = p_wind_mw * cuf_w * 8760.0 / 1000.0
    p50_gwh = e_solar_gwh + e_wind_gwh
    p90_gwh = p50_gwh * 0.87               # 1.28-sigma inter-annual variability
    cuf_combined = p50_gwh * 1000.0 / max(cap_mw * 8.760, 1e-9) / 1000.0

    capex_cr = (p_solar_mwp * SOLAR_CAPEX_CR_PER_MW
                + p_wind_mw * WIND_CAPEX_CR_PER_MW
                + tx_line_km * TX_LINE_CR_PER_KM)
    om_cr_yr = capex_cr * OM_PCT_OF_CAPEX

    # 25-year cashflows in INR crore (1 GWh @ tariff INR/kWh = tariff * 0.1 crore)
    cash = [-capex_cr]
    disc_energy = 0.0
    disc_cost = capex_cr
    for yr in range(1, PROJECT_LIFE_YEARS + 1):
        e_yr = p50_gwh * (1 - DEGRADATION_PCT_YR) ** (yr - 1)
        rev = e_yr * PPA_TARIFF_INR_KWH * 0.1
        cash.append(rev - om_cr_yr)
        disc_energy += e_yr * 1e6 / (1 + DISCOUNT_RATE) ** yr      # kWh
        disc_cost += om_cr_yr / (1 + DISCOUNT_RATE) ** yr
    npv_cr = sum(cf / (1 + DISCOUNT_RATE) ** t for t, cf in enumerate(cash))
    lcoe = disc_cost * 1e7 / max(disc_energy, 1e-9)                 # INR/kWh
    irr_pct = irr(cash) * 100.0

    return {
        "installedSolarMwp": round(p_solar_mwp, 1),
        "installedWindMw": round(p_wind_mw, 1),
        "turbineUnitMw": TURBINE_MW,
        "cufSolarPct": round(cuf_s * 100, 1),
        "cufWindPct": round(cuf_w * 100, 1),
        "cufCombinedPct": round(cuf_combined * 100, 1),
        "p50NetGenerationGwhYr": round(p50_gwh, 1),
        "p90NetGenerationGwhYr": round(p90_gwh, 1),
        "capexCr": round(capex_cr, 1),
        "txLineKm": round(tx_line_km, 1),
        "txLineCostCr": round(tx_line_km * TX_LINE_CR_PER_KM, 2),
        "omCrYr": round(om_cr_yr, 2),
        "ppaTariffInrKwh": PPA_TARIFF_INR_KWH,
        "lcoeInrKwh": round(lcoe, 2),
        "npvCr": round(npv_cr, 1),
        "irrPct": round(irr_pct, 2),
        "projectLifeYears": PROJECT_LIFE_YEARS,
        "discountRatePct": round(DISCOUNT_RATE * 100, 1),
    }
