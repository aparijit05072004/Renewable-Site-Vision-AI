"""ParcelMonitoringState — the durable multi-agent state (blueprint §8.2)."""
from __future__ import annotations

from typing import Optional, TypedDict


class ParcelMonitoringState(TypedDict, total=False):
    parcelId: str
    messages: list                      # agent log, append-only
    netBuildableHa: float
    buildableRatioPct: float
    baselineCloudCover: float
    currentCloudCover: float
    baselineWpd120m: float
    currentWpd120m: float
    baselineNdvi: float
    currentNdvi: float
    baselineNdbi: float
    currentNdbi: float
    meteoDriftDetected: bool
    orbitalEncroachmentDetected: bool
    zoningViolationDetected: bool
    suitabilityScore: float
    projectedIrrPct: float
    minRoiThresholdPct: float
    stakeholderDecision: Optional[str]  # approve | abandon | None (awaiting HITL)
    hitlPending: bool
    cycle: int


def initial_state(parcel_id: str, buildable_ha: float, buildable_pct: float,
                  cloud: float, wpd: float, irr_pct: float,
                  suitability: float, min_roi: float = 9.5) -> ParcelMonitoringState:
    return ParcelMonitoringState(
        parcelId=parcel_id, messages=[], netBuildableHa=buildable_ha,
        buildableRatioPct=buildable_pct, baselineCloudCover=cloud,
        currentCloudCover=cloud, baselineWpd120m=wpd, currentWpd120m=wpd,
        baselineNdvi=0.34, currentNdvi=0.34, baselineNdbi=-0.08,
        currentNdbi=-0.08, meteoDriftDetected=False,
        orbitalEncroachmentDetected=False, zoningViolationDetected=False,
        suitabilityScore=suitability, projectedIrrPct=irr_pct,
        minRoiThresholdPct=min_roi, stakeholderDecision=None,
        hitlPending=False, cycle=0)
