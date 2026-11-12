"""TERRA — the TerraGuard-v2 site-intelligence assistant.

A report-grounded conversational engine for the dashboard.  Two tiers:

1. **Grounded deterministic core (always available, offline):** intent
   matching over the live 6-section report JSON plus a renewable-energy
   domain knowledge base.  Every numeric answer is read from the actual
   analysis — never invented.
2. **Claude API tier (optional):** if the ``anthropic`` SDK is installed
   and ``ANTHROPIC_API_KEY`` is set, free-form questions are answered by
   ``claude-opus-5-5`` with the report JSON supplied as grounding context
   (server-side refusal fallback enabled).  Absent a key, tier 1 answers
   everything.
"""
from __future__ import annotations

import os
import re

_KNOWLEDGE = {
    "wpd": ("Wind Power Density (WPD) = 0.5 * rho * v^3 [W/m^2]. TerraGuard "
            "computes it at 120 m hub height using Aurora-style dynamic air "
            "density and CorrDiff-downscaled 2 km winds; >= 400 W/m^2 is "
            "generally bankable for utility wind."),
    "weibull": ("The Weibull distribution models wind-speed frequency with "
                "shape k (~2 for inland India) and scale A. P50 energy "
                "estimates integrate the turbine power curve over it."),
    "corrdiff": ("CorrDiff is NVIDIA's two-step generative downscaler: a "
                 "deterministic regression U-Net predicts the 2 km "
                 "conditional mean, then an EDM diffusion model restores "
                 "the stochastic k^-5/3 turbulence spectrum that MSE "
                 "training smooths away."),
    "pinn": ("The ABL-PINN solves steady Navier-Stokes + Ekman-balance PDEs "
             "over the 30 m DEM with the image-derived roughness z0(x,y) as "
             "a no-slip boundary, resolving ridgeline speed-up and hub-"
             "height shear that data-driven models miss."),
    "sda": ("Score-Based Data Assimilation fuses sparse NIWE 120 m mast "
            "observations into coarse ERA5/GraphCast fields via guided "
            "reverse-time diffusion, correcting systematic regional bias "
            "before downscaling."),
    "gnn": ("The Fuzzy Context-Enriched Spatial GNN replaces AHP weighting. "
            "Domain-anchored sigmoid memberships normalise each criterion "
            "independently of the candidate pool, so rankings are immune "
            "to the AHP Rank Reversal paradox; GATv2 attention then mixes "
            "regional context."),
    "lcoe": ("Levelized Cost of Energy = discounted lifetime cost / "
             "discounted lifetime generation. India utility-solar LCOE is "
             "roughly INR 2.5-3.5/kWh; wind INR 2.8-3.8/kWh."),
    "cuf": ("Capacity Utilisation Factor = actual generation / (nameplate x "
            "8760 h). Indian solar runs ~17-23%, good wind sites 28-38%."),
    "gcr": ("Ground Coverage Ratio (~0.45 here) is the panel-area to land-"
            "area ratio; with ~1.6 ha/MWp it sets installable solar MWp."),
    "setback": ("Regulatory setbacks applied by Layer 0: habitation 150 m, "
                "forest canopy 100 m, water/wetland 200 m, roads 15 m — "
                "dilated around obstacle pixels before computing buildable "
                "area."),
    "hitl": ("Human-in-the-loop governance: when re-projected IRR drops "
             "below the investor floor, the LangGraph-style workflow "
             "interrupts and persists; a stakeholder resumes it via "
             "POST /api/v1/hitl/resume with approve/abandon."),
}

_SMALL_TALK = {
    "hello": "Hello! I'm TERRA ⚡ — your renewable site-intelligence copilot. "
             "Upload a land image or ask me anything about this site's wind, "
             "solar, suitability or financials.",
    "help": "Try: 'summarize this site', 'why this suitability score?', "
            "'wind or solar here?', 'explain the financials', 'what is "
            "CorrDiff?', 'monitoring status', or 'is it worth investing?'",
    "thanks": "Anytime! May your capacity factors be ever in your favour ☀️",
}


class TerraAssistant:
    def __init__(self):
        self._anthropic = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # type: ignore
                self._anthropic = anthropic.Anthropic()
            except Exception:
                self._anthropic = None

    # ------------------------------------------------------------------
    def chat(self, message: str, report: dict | None) -> dict:
        grounded = self._grounded_answer(message, report)
        if grounded is not None:
            return {"reply": grounded, "tier": "grounded"}
        if self._anthropic is not None and report is not None:
            llm = self._claude_answer(message, report)
            if llm:
                return {"reply": llm, "tier": "claude"}
        return {"reply": self._fallback(message, report), "tier": "grounded"}

    # ------------------------------------------------------------------
    def _grounded_answer(self, message: str, report: dict | None) -> str | None:
        q = message.lower().strip()
        for key, reply in _SMALL_TALK.items():
            if re.search(rf"\b{key}|^(hi|hey)\b" if key == "hello" else rf"\b{key}", q):
                return reply
        for key, text in _KNOWLEDGE.items():
            if re.search(rf"\b{key}\b|rank reversal" if key == "gnn" else rf"\b{key}\b", q):
                return text
        if report is None:
            if any(w in q for w in ("site", "score", "report", "summary",
                                    "financ", "invest", "wind", "solar")):
                return ("No site has been analysed yet in this session — drop "
                        "a land image (GeoTIFF, drone JPEG, or aerial "
                        "screenshot) on the upload pad and I'll run the full "
                        "Layer 0-4 cascade for you.")
            return None

        a, b = report["sectionA"], report["sectionB"]
        c, d = report["sectionC"], report["sectionD"]
        e = report["sectionE"]
        if any(w in q for w in ("summar", "overview", "tell me about",
                                "describe")):
            return (f"📍 Site verdict: Grade {c['grade']} "
                    f"({c['suitabilityScore100']}/100), best suited for "
                    f"{c['recommendedTechnology']}. Of "
                    f"{a['totalFootprintHa']} ha, {a['netBuildableHa']} ha "
                    f"({a['netBuildablePct']}%) is buildable after setbacks. "
                    f"Wind: {b['niweMeanWind120m']} m/s @120 m "
                    f"(WPD {b['pinnWpd120mWm2']} W/m²); solar G_eff "
                    f"{b['geffKwhM2Day']} kWh/m²/day. Plan: "
                    f"{d['installedSolarMwp']} MWp solar + "
                    f"{d['installedWindMw']} MW wind, P50 "
                    f"{d['p50NetGenerationGwhYr']} GWh/yr, IRR "
                    f"{d['irrPct']}%, LCOE ₹{d['lcoeInrKwh']}/kWh.")
        if "score" in q or "suitab" in q or "grade" in q or "why" in q:
            return (f"The fuzzy spatial GNN scored this parcel "
                    f"{c['suitabilityScore100']}/100 (Grade {c['grade']}, "
                    f"{c['regionalPercentile']}th percentile regionally). "
                    f"Drivers — wind membership {c['windMembership']}, solar "
                    f"membership {c['solarMembership']}, buildable ratio "
                    f"{a['netBuildablePct']}%, substation "
                    f"{c['nearestSubstationKm']} km, highway "
                    f"{c['nearestHighwayKm']} km. The score is rank-"
                    f"invariant: adding or removing other candidate sites "
                    f"cannot change it.")
        if ("wind" in q and "solar" in q) or "hybrid" in q or "technology" in q \
                or "recommend" in q:
            return (f"Recommended configuration: {c['recommendedTechnology']}. "
                    f"Wind case — {b['niweMeanWind120m']} m/s mean @120 m, "
                    f"Weibull A={b['weibull']['A']}, k={b['weibull']['k']}, "
                    f"WPD {b['pinnWpd120mWm2']} W/m² with "
                    f"{b['ridgelineSpeedupPct']}% ridgeline speed-up. Solar "
                    f"case — G_eff {b['geffKwhM2Day']} kWh/m²/day after "
                    f"{b['annualSoilingAttenuationPct']}% PM soiling loss. "
                    f"CUFs: wind {d['cufWindPct']}%, solar {d['cufSolarPct']}%.")
        if "wind" in q:
            return (f"Wind resource: SDA-calibrated mean {b['niweMeanWind120m']}"
                    f" m/s at 120 m (Weibull A={b['weibull']['A']}, "
                    f"k={b['weibull']['k']}), air density "
                    f"{b['airDensityKgM3']} kg/m³, PINN WPD "
                    f"{b['pinnWpd120mWm2']} W/m² (peak {b['pinnMaxWpdWm2']}), "
                    f"ridgeline speed-up {b['ridgelineSpeedupPct']}%. Layout "
                    f"fits {a['plannedTurbines']} × 3.3 MW turbines at 7D "
                    f"wake spacing.")
        if "solar" in q or "irradiance" in q or "pv" in q:
            return (f"Solar resource: GHI {b['ghiKwhM2Day']} kWh/m²/day → "
                    f"G_eff {b['geffKwhM2Day']} after cloud "
                    f"({b['meanCloudCover']}), bifacial albedo gain and "
                    f"{b['annualSoilingAttenuationPct']}% PM2.5/PM10 soiling. "
                    f"Buildable land supports {a['plannedSolarMwp']} MWp at "
                    f"GCR 0.45.")
        if any(w in q for w in ("financ", "invest", "irr", "npv", "capex",
                                "lcoe", "worth", "money", "cost", "revenue")):
            verdict = ("a bankable configuration" if d["irrPct"] >= 12 else
                       "viable but thin — negotiate the PPA" if d["irrPct"] >= 9.5
                       else "below typical investor floors")
            return (f"25-year economics: CAPEX ₹{d['capexCr']} Cr (incl. "
                    f"₹{d['txLineCostCr']} Cr for {d['txLineKm']} km "
                    f"evacuation line), P50 {d['p50NetGenerationGwhYr']} "
                    f"GWh/yr (P90 {d['p90NetGenerationGwhYr']}), LCOE "
                    f"₹{d['lcoeInrKwh']}/kWh vs PPA ₹{d['ppaTariffInrKwh']}, "
                    f"NPV ₹{d['npvCr']} Cr @ {d['discountRatePct']}% discount, "
                    f"IRR {d['irrPct']}% — {verdict}.")
        if any(w in q for w in ("buildable", "area", "land", "parcel",
                                "footprint", "class")):
            top = sorted(a["classDistributionPct"].items(),
                         key=lambda kv: -kv[1])[:3]
            mix = ", ".join(f"{k} {v}%" for k, v in top)
            return (f"Land audit: {a['totalFootprintHa']} ha total, "
                    f"{a['netBuildableHa']} ha ({a['netBuildablePct']}%) "
                    f"buildable after setbacks. Dominant cover: {mix}. Mean "
                    f"roughness z0={a['meanRoughnessZ0']} m, albedo "
                    f"{a['meanSurfaceAlbedo']}, road access: "
                    f"{'yes' if a['internalRoadAccess'] else 'no'}.")
        if any(w in q for w in ("monitor", "drift", "agent", "thread",
                                "checkpoint", "risk")):
            return (f"Monitoring thread {e['monitoringThreadId']} is live with "
                    f"durable checkpointing ({e['checkpointBackend']}). "
                    f"Triggers: cloud drift ≥{e['cloudDriftTriggerPct']}%, WPD "
                    f"drift ≥{e['wpdDriftTriggerPct']}%, ΔNDVI "
                    f">{e['ndviDriftTrigger']}, ΔNDBI >{e['ndbiDriftTrigger']}; "
                    f"Sentinel-2 revisit every {e['revisitCadenceDays']} days. "
                    f"If re-projected IRR breaches the "
                    f"{report['sectionF']['irrFloorPct']}% floor, the workflow "
                    f"interrupts for human sign-off.")
        if "grid" in q or "substation" in q or "evacuat" in q or "highway" in q:
            return (f"Grid & logistics: nearest ≥66 kV substation "
                    f"{c['nearestSubstation']['name']} "
                    f"({c['nearestSubstation']['voltageKv']} kV) at "
                    f"{c['nearestSubstationKm']} km; nearest highway "
                    f"{c['nearestHighwayKm']} km. Evacuation line cost ≈ "
                    f"₹{d['txLineCostCr']} Cr.")
        return None

    # ------------------------------------------------------------------
    def _claude_answer(self, message: str, report: dict) -> str | None:
        import json
        try:
            resp = self._anthropic.beta.messages.create(
                model="claude-opus-5-5",
                max_tokens=1024,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=("You are TERRA, the site-intelligence assistant of "
                        "TerraGuard-v2, a renewable-energy site selection "
                        "platform. Answer concisely and only from the "
                        "analysis report JSON provided; if the answer is not "
                        "derivable from it, say so."),
                messages=[{"role": "user", "content":
                           f"<report>{json.dumps(report)}</report>\n\n"
                           f"{message}"}])
            if resp.stop_reason == "refusal":
                return None
            return "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text") or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback(message: str, report: dict | None) -> str:
        hint = ("Ask about: site summary, suitability score, wind/solar "
                "resource, financials (IRR, NPV, LCOE), buildable land, grid "
                "distance, monitoring & drift agents — or concepts like WPD, "
                "Weibull, CorrDiff, PINN, SDA, GNN, CUF, LCOE.")
        if report is None:
            return ("I couldn't match that. Upload a land image first, then "
                    + hint)
        return "I couldn't match that to the report. " + hint
