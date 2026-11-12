"""FastAPI Image-to-Report & Multi-Agent service (blueprint §8.2).

Endpoints
---------
POST /api/v1/analyze-land-image   land image -> full 6-section report
POST /api/v1/hitl/resume          stakeholder sign-off (Command resume)
POST /api/v1/monitor/step         advance one monitoring cycle
POST /api/v1/assistant/chat       TERRA assistant
GET  /api/v1/report/{thread_id}   report JSON
GET  /api/v1/report/{thread_id}/pdf
GET  /api/v1/threads              monitoring threads
Static: serves dashboard-ui/dist when built.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import DATA_DIR, REPO_ROOT
from ..layer0_vision.land_cover_spec import CLASS_COLORS, CLASS_NAMES
from ..layer0_vision.layout_overlay_renderer import (
    render_layout_overlay, render_segmentation_overlay)
from ..pipeline import TerraGuardPipeline
from .assistant import TerraAssistant
from .persistent_workflow import PersistentParcelWorkflow
from .report_synthesizer import report_to_pdf, synthesize_report

app = FastAPI(title="TerraGuard-v2 Image-to-Report & Multi-Agent API",
              version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_pipeline: TerraGuardPipeline | None = None
_workflow = PersistentParcelWorkflow()
_assistant = TerraAssistant()
_REPORT_STORE = DATA_DIR / "reports"
_REPORT_STORE.mkdir(parents=True, exist_ok=True)


def get_pipeline() -> TerraGuardPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TerraGuardPipeline()
    return _pipeline


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "engine": "TerraGuard-v2"}


@app.post("/api/v1/analyze-land-image")
async def analyze_land_image(file: UploadFile = File(...),
                             pinLat: float | None = Form(None),
                             pinLon: float | None = Form(None),
                             gsdM: float | None = Form(None)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    pin = (pinLat, pinLon) if pinLat is not None and pinLon is not None else None
    try:
        analysis = get_pipeline().analyze(data, file.filename or "upload.png",
                                          pin_latlon=pin, gsd_override=gsdM)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(422, f"analysis failed: {exc}") from exc

    l0 = analysis["layer0"]
    # register persistent monitoring thread
    thread_id = _workflow.register_parcel({
        "netBuildableHa": l0["netBuildableHa"],
        "netBuildablePct": l0["netBuildablePct"],
        "meanCloudCover": analysis["atmosphere"]["meanCloudCover"],
        "meanWpd120m": analysis["atmosphere"]["meanWpd120m"],
        "irrPct": 0.0,   # filled after financials below
        "suitabilityScore": analysis["gnn"]["suitabilityScore"],
    })
    report = synthesize_report(analysis, thread_id)
    # sync projected IRR into the monitoring state
    state = _workflow.saver.get(thread_id)
    state["projectedIrrPct"] = report["sectionD"]["irrPct"]
    _workflow.saver.put(thread_id, state)

    # overlays
    from PIL import Image
    import io as _io
    rgb = np.asarray(Image.open(_io.BytesIO(data)).convert("RGB"))
    seg_png = render_segmentation_overlay(rgb, l0["segmentation"], CLASS_COLORS)
    layout_png = render_layout_overlay(rgb, l0["buildableMask"],
                                       l0["anchor"]["gsd_m"], l0["nTurbines"])

    payload = {
        "report": report,
        "overlays": {
            "segmentationPng": base64.b64encode(seg_png).decode(),
            "layoutPng": base64.b64encode(layout_png).decode(),
            "classNames": CLASS_NAMES,
            "classColors": CLASS_COLORS,
        },
        "heatmap": {
            "wpd": _grid_to_list(analysis["grids"]["wpdGrid"]),
            "geff": _grid_to_list(analysis["grids"]["geffGrid"]),
            "dem": _grid_to_list(analysis["grids"]["demGrid"]),
        },
    }
    (_REPORT_STORE / f"{thread_id}.json").write_text(json.dumps(report))
    return payload


@app.get("/api/v1/report/{thread_id}")
def get_report(thread_id: str) -> dict:
    p = _REPORT_STORE / f"{thread_id}.json"
    if not p.exists():
        raise HTTPException(404, "unknown report")
    return json.loads(p.read_text())


@app.get("/api/v1/report/{thread_id}/pdf")
def get_report_pdf(thread_id: str) -> Response:
    p = _REPORT_STORE / f"{thread_id}.json"
    if not p.exists():
        raise HTTPException(404, "unknown report")
    pdf = report_to_pdf(json.loads(p.read_text()))
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition":
            f'attachment; filename="terraguard_{thread_id}.pdf"'})


@app.post("/api/v1/monitor/step")
def monitor_step(body: dict) -> dict:
    thread_id = body.get("threadId")
    if not thread_id:
        raise HTTPException(400, "threadId required")
    try:
        state = _workflow.step(thread_id, body.get("sensorUpdate"))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"state": state}


@app.post("/api/v1/hitl/resume")
def hitl_resume(body: dict) -> dict:
    thread_id = body.get("threadId")
    decision = body.get("decision")
    if decision not in ("approve", "abandon"):
        raise HTTPException(400, "decision must be approve|abandon")
    try:
        state = _workflow.resume_hitl(thread_id, decision,
                                      body.get("overrideThresholds"))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"state": state}


@app.get("/api/v1/threads")
def list_threads() -> dict:
    out = []
    for tid in _workflow.saver.list_threads():
        st = _workflow.saver.get(tid)
        out.append({"threadId": tid, "cycle": st.get("cycle", 0),
                    "irrPct": st.get("projectedIrrPct"),
                    "hitlPending": st.get("hitlPending", False),
                    "decision": st.get("stakeholderDecision")})
    return {"threads": out}


@app.post("/api/v1/assistant/chat")
def assistant_chat(body: dict) -> dict:
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "message required")
    report = None
    tid = body.get("threadId")
    if tid:
        p = _REPORT_STORE / f"{tid}.json"
        if p.exists():
            report = json.loads(p.read_text())
    return _assistant.chat(message, report)


def _grid_to_list(grid: np.ndarray, n: int = 48) -> list:
    import torch
    t = torch.from_numpy(np.ascontiguousarray(grid, dtype=np.float32))[None, None]
    small = torch.nn.functional.interpolate(t, (n, n), mode="bilinear",
                                            align_corners=False)[0, 0]
    return [[round(float(v), 2) for v in row] for row in small]


# ---- static dashboard (after API routes) ----
_dist = REPO_ROOT / "dashboard-ui" / "dist"
if _dist.exists():  # pragma: no cover
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")
