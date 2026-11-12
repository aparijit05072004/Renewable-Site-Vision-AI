"""Multi-agent nodes of the cyclic monitoring graph (blueprint §8.2).

Meteorological Drift Agent  — 10-year cloud/WPD drift margins
Geospatial & Regulatory Agent — Sentinel-2 NDVI/NDBI orbital encroachment
Financial Orchestrator Agent — IRR re-projection under drift penalties
HITL Gate                    — interrupt() when IRR falls below threshold
"""
from __future__ import annotations

from .state import ParcelMonitoringState


def meteorological_drift_node(state: ParcelMonitoringState) -> dict:
    cloud_delta = ((state["currentCloudCover"] - state["baselineCloudCover"])
                   / max(state["baselineCloudCover"], 1e-4))
    wpd_drop = ((state["baselineWpd120m"] - state["currentWpd120m"])
                / max(state["baselineWpd120m"], 1e-4))
    drift = (cloud_delta >= 0.05) or (wpd_drop >= 0.04)
    msg = (f"[MeteoDriftAgent] Cloud shift={cloud_delta:+.2%}, "
           f"WPD drop={wpd_drop:+.2%}, Drift={drift}")
    return {"meteoDriftDetected": drift,
            "messages": [{"role": "assistant", "content": msg}]}


def geospatial_regulatory_node(state: ParcelMonitoringState) -> dict:
    ndvi_drop = state["baselineNdvi"] - state["currentNdvi"]
    ndbi_rise = state["currentNdbi"] - state["baselineNdbi"]
    encroach = (ndvi_drop > 0.08) or (ndbi_rise > 0.06)
    zoning = state.get("zoningViolationDetected", False)
    msg = (f"[GeoRegulatoryAgent] Sentinel-2 dNDVI={-ndvi_drop:+.3f}, "
           f"dNDBI={ndbi_rise:+.3f}, Encroachment={encroach}, Zoning={zoning}")
    return {"orbitalEncroachmentDetected": encroach,
            "messages": [{"role": "assistant", "content": msg}]}


def financial_orchestrator_node(state: ParcelMonitoringState) -> dict:
    irr = state["projectedIrrPct"]
    if (state["meteoDriftDetected"] or state["orbitalEncroachmentDetected"]
            or state["zoningViolationDetected"]):
        drift_penalty = ((0.14 if state["meteoDriftDetected"] else 0.0)
                         + (0.28 if state["orbitalEncroachmentDetected"] else 0.0)
                         + (0.35 if state["zoningViolationDetected"] else 0.0))
        irr = irr * (1.0 - drift_penalty)
    msg = f"[FinancialAgent] Re-projected IRR={irr:.2f}% (floor {state['minRoiThresholdPct']:.1f}%)"
    return {"projectedIrrPct": round(irr, 2),
            "messages": [{"role": "assistant", "content": msg}]}


def hitl_gate_node(state: ParcelMonitoringState) -> dict:
    """Pause the graph for human sign-off when IRR breaches the floor."""
    if state["projectedIrrPct"] < state["minRoiThresholdPct"] \
            and state.get("stakeholderDecision") is None:
        msg = ("[HITLGate] IRR below investor floor — interrupt() raised; "
               "awaiting stakeholder decision via /api/v1/hitl/resume")
        return {"hitlPending": True,
                "messages": [{"role": "assistant", "content": msg}]}
    return {"hitlPending": False, "messages": []}


def route_after_hitl(state: ParcelMonitoringState) -> str:
    if state.get("hitlPending"):
        return "WAIT_HUMAN"
    if state.get("stakeholderDecision") == "abandon":
        return "END"
    return "CONTINUE"
