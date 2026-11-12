"""6-Section Renewable Energy Site Selection Report generator
(blueprint §8.1): interactive JSON for the React UI + downloadable PDF.
"""
from __future__ import annotations

import datetime as _dt
import io

from ..layer3_spatial.context_spatial_gnn import FEATURES  # noqa: F401
from .financials import project_financials


def synthesize_report(analysis: dict, thread_id: str) -> dict:
    l0 = analysis["layer0"]
    atm = analysis["atmosphere"]
    gnn = analysis["gnn"]
    excl = analysis["exclusions"]
    fin = project_financials(
        p_solar_mwp=l0["pSolarMwp"], p_wind_mw=l0["pWindMw"],
        geff_kwh_m2_day=atm["geffKwhM2Day"],
        mean_speed_120m=atm["meanWindSpeed120m"],
        tx_line_km=excl["nearestSubstationKm"])

    report = {
        "reportId": thread_id,
        "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "engine": "TerraGuard-v2",
        "anchor": analysis["anchor"],
        "inferenceSeconds": analysis["inferenceSeconds"],
        "sectionA": {
            "title": "Visual Land Parcel Audit & Annotated Layout Blueprint",
            "totalFootprintHa": l0["totalFootprintHa"],
            "netBuildableHa": l0["netBuildableHa"],
            "netBuildablePct": l0["netBuildablePct"],
            "classDistributionPct": l0["classDistributionPct"],
            "setbacksAppliedM": l0["setbacksAppliedM"],
            "internalRoadAccess": l0["roadAccess"],
            "meanRoughnessZ0": round(l0["meanZ0"], 4),
            "meanSurfaceAlbedo": round(l0["meanAlbedo"], 3),
            "plannedSolarMwp": l0["pSolarMwp"],
            "plannedTurbines": l0["nTurbines"],
        },
        "sectionB": {
            "title": "Cascading Micro-Climate & Physics Profile",
            "sdaCalibration": analysis["sda"],
            "niweMeanWind120m": atm["meanWindSpeed120m"],
            "weibull": {"A": atm["weibullA"], "k": atm["weibullK"]},
            "airDensityKgM3": atm["airDensity"],
            "pinnWpd120mWm2": atm["meanWpd120m"],
            "pinnMaxWpdWm2": atm["maxWpd120m"],
            "ridgelineSpeedupPct": atm["ridgelineSpeedupPct"],
            "ghiKwhM2Day": atm["ghiKwhM2Day"],
            "geffKwhM2Day": atm["geffKwhM2Day"],
            "meanCloudCover": atm["meanCloudCover"],
            "pm25UgM3": atm["pm25"], "pm10UgM3": atm["pm10"],
            "annualSoilingAttenuationPct": atm["soilingAttenuationPct"],
            "terrainSlopeDeg": atm["terrainSlopeDeg"],
        },
        "sectionC": {
            "title": "Fuzzy Spatial GNN Suitability & Technology Recommendation",
            "suitabilityScore100": gnn["suitabilityScore100"],
            "grade": gnn["grade"],
            "recommendedTechnology": gnn["recommendedTechnology"],
            "windMembership": gnn["windMembership"],
            "solarMembership": gnn["solarMembership"],
            "regionalPercentile": gnn["regionalPercentile"],
            "nearestSubstationKm": excl["nearestSubstationKm"],
            "nearestSubstation": excl["nearestSubstation"],
            "nearestHighwayKm": excl["nearestHighwayKm"],
            "parcelPermissible": excl["parcelPermissible"],
            "exclusionBuffersM": excl["exclusionBuffersM"],
        },
        "sectionD": {"title": "25-Year Financial Viability & Energy Yield",
                     **fin},
        "sectionE": {
            "title": "Multi-Agent Continuous Drift & Regulatory Risk Ledger",
            "monitoringThreadId": thread_id,
            "checkpointBackend": "SqliteSaver (PostgresSaver-compatible)",
            "cloudDriftTriggerPct": 5.0,
            "wpdDriftTriggerPct": 4.0,
            "ndviDriftTrigger": 0.08,
            "ndbiDriftTrigger": 0.06,
            "revisitCadenceDays": 5,
            "tenYearCloudDriftMarginPct": round(
                5.0 * (1 - analysis["atmosphere"]["meanCloudCover"]), 2),
            "tenYearWpdDriftMarginPct": 4.0,
        },
        "sectionF": {
            "title": "Human-in-the-Loop Governance & Sign-Off Console",
            "hitlEndpoint": "/api/v1/hitl/resume",
            "irrFloorPct": 9.5,
            "overridableThresholds": ["minRoiThresholdPct", "cloudDriftTriggerPct",
                                      "wpdDriftTriggerPct"],
            "status": "ARMED",
        },
    }
    return report


# ----------------------------------------------------------------------
# PDF rendering (reportlab)
# ----------------------------------------------------------------------
def report_to_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm,
                            title="TerraGuard-v2 Site Selection Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=17,
                        textColor=colors.HexColor("#0b3d2e"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        textColor=colors.HexColor("#0e7490"), spaceBefore=10)
    body = styles["BodyText"]

    def kv_table(d: dict) -> Table:
        rows = [[str(k), str(v)] for k, v in d.items()
                if not isinstance(v, (dict, list)) and k != "title"]
        t = Table(rows, colWidths=[72 * mm, 92 * mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#334155")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#f1f5f9")]),
        ]))
        return t

    story = [Paragraph("TerraGuard-v2 — Renewable Energy Site Selection "
                       "Report", h1),
             Paragraph(f"Report {report['reportId']} · generated "
                       f"{report['generatedAt'][:19]} UTC · anchor "
                       f"({report['anchor']['lat']:.4f}, "
                       f"{report['anchor']['lon']:.4f}) · end-to-end "
                       f"inference {report['inferenceSeconds']} s", body),
             Spacer(1, 6)]
    for key, label in (("sectionA", "A"), ("sectionB", "B"), ("sectionC", "C"),
                       ("sectionD", "D"), ("sectionE", "E"), ("sectionF", "F")):
        sec = report[key]
        story.append(Paragraph(f"Section {label} — {sec['title']}", h2))
        story.append(kv_table(sec))
    doc.build(story)
    return buf.getvalue()
