"""Phase 5 — End-to-End & Persistence gate (blueprint Table 4).

1. Upload a raw land image -> full 6-section report in < 6 s.
2. Simulate a +6% cloud-drift interrupt, destroy the in-memory workflow
   (container-restart analogue), rebuild it from the durable checkpoint
   store, and verify lossless state resumption.
Also consolidates every layer's gate metrics into one validation report.
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
from PIL import Image

from ..config import METRICS_DIR
from ..layer0_vision.synthetic_tiles import generate_tile
from ..layer4_orchestrator.persistent_workflow import (PersistentParcelWorkflow,
                                                       SqliteSaver)
from ..layer4_orchestrator.report_synthesizer import (report_to_pdf,
                                                      synthesize_report)
from ..pipeline import TerraGuardPipeline


def main() -> dict:
    # ---------- E2E image -> report latency ----------
    rgb, _ = generate_tile(512, seed=4242)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    pipeline = TerraGuardPipeline()          # cold model load (excluded)
    t0 = time.perf_counter()
    analysis = pipeline.analyze(img_bytes, "validation_parcel.png",
                                pin_latlon=(14.2250, 76.3980), gsd_override=10.0)
    wf = PersistentParcelWorkflow()
    thread_id = wf.register_parcel({
        "netBuildableHa": analysis["layer0"]["netBuildableHa"],
        "netBuildablePct": analysis["layer0"]["netBuildablePct"],
        "meanCloudCover": analysis["atmosphere"]["meanCloudCover"],
        "meanWpd120m": analysis["atmosphere"]["meanWpd120m"],
        "irrPct": 11.0,
        "suitabilityScore": analysis["gnn"]["suitabilityScore"]})
    report = synthesize_report(analysis, thread_id)
    pdf_bytes = report_to_pdf(report)
    e2e_seconds = time.perf_counter() - t0

    # ---------- Drift interrupt + lossless restart ----------
    base_cloud = analysis["atmosphere"]["meanCloudCover"]
    wf.step(thread_id, sensor_update={
        "currentCloudCover": base_cloud * 1.06,     # +6% drift
        "projectedIrrPct": 8.9})                     # below 9.5% floor
    state_before = wf.saver.get(thread_id)
    interrupted = bool(state_before["hitlPending"])

    del wf                                           # simulate container loss
    wf2 = PersistentParcelWorkflow(SqliteSaver())    # fresh process, same DB
    state_after = wf2.saver.get(thread_id)
    lossless = state_after == state_before
    resumed = wf2.resume_hitl(thread_id, "approve")
    resume_ok = (resumed["stakeholderDecision"] == "approve"
                 and not resumed["hitlPending"]
                 and resumed["cycle"] == state_before["cycle"])

    # ---------- consolidate ----------
    layers = {}
    for name in ("layer0_segmentation", "layer1_sda", "layer2_corrdiff",
                 "layer2_pinn", "layer3_gnn"):
        p = METRICS_DIR / f"{name}.json"
        if p.exists():
            layers[name] = json.loads(p.read_text())

    metrics = {
        "e2eSeconds": round(e2e_seconds, 2),
        "reportSections": [k for k in report if k.startswith("section")],
        "pdfBytes": len(pdf_bytes),
        "suitabilityScore100": report["sectionC"]["suitabilityScore100"],
        "driftInterruptRaised": interrupted,
        "losslessStateResumption": lossless,
        "hitlResumeOk": resume_ok,
        "gate": {
            "e2e<6s": e2e_seconds < 6.0,
            "driftInterrupt": interrupted,
            "losslessResume": lossless and resume_ok,
        },
        "layerGates": {k: v.get("gate") for k, v in layers.items()},
    }
    all_gates = all(metrics["gate"].values()) and all(
        all(g.values()) for g in metrics["layerGates"].values() if g)
    metrics["allGatesPassed"] = all_gates
    (METRICS_DIR / "validation_suite.json").write_text(
        json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
