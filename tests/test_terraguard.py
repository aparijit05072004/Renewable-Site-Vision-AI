"""TerraGuard-v2 test suite (pytest)."""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from terraguard.common.fields import psd_slope, spectral_field, synthetic_dem
from terraguard.layer0_vision.geo_metadata_resolver import resolve_image_geolocation
from terraguard.layer0_vision.land_cover_spec import NUM_CLASSES
from terraguard.layer0_vision.land_image_analyzer import LandImagePerceptionEngine
from terraguard.layer0_vision.synthetic_tiles import generate_tile
from terraguard.layer1_sda.score_network import ScoreUNet, cosine_alpha_bars
from terraguard.layer1_sda.sda_latent_assimilator import ScoreBasedDataAssimilator
from terraguard.layer2_atmosphere.corrdiff_downscaler import CorrDiffCascadedDownscaler
from terraguard.layer3_spatial.context_spatial_gnn import (
    ContextEnrichedSpatialGNN, knn_spatial_graph)
from terraguard.layer4_orchestrator.financials import irr, project_financials
from terraguard.layer4_orchestrator.persistent_workflow import (
    PersistentParcelWorkflow, SqliteSaver)


def _png_bytes(n=128, seed=1):
    rgb, _ = generate_tile(n, seed)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


# ---------------- common ----------------
def test_spectral_field_slope():
    rng = np.random.default_rng(0)
    f = spectral_field(128, -5.0 / 3.0, rng)
    assert abs(psd_slope(f) - (-5.0 / 3.0)) < 0.35


def test_synthetic_dem_range():
    dem = synthetic_dem(64, np.random.default_rng(1))
    assert 500 < dem.mean() < 1100 and dem.min() >= 600 - 1


# ---------------- layer 0 ----------------
def test_geo_resolver_fallback_pin():
    anchor = resolve_image_geolocation(_png_bytes(), "shot.png",
                                       fallback_latlon=(12.5, 77.1))
    assert anchor.source == "map_pin"
    assert anchor.lat == pytest.approx(12.5)


def test_layer0_analysis_shapes():
    eng = LandImagePerceptionEngine()
    out = eng.analyze(_png_bytes(160, 7), "tile.png", gsd_override=10.0)
    seg = out["segmentation"]
    assert seg.shape == (160, 160)
    assert seg.max() < NUM_CLASSES
    assert out["netBuildableHa"] <= out["totalFootprintHa"]
    assert 0 <= out["netBuildablePct"] <= 100
    assert out["z0Grid"].shape == seg.shape


# ---------------- layer 1 ----------------
def test_sda_observation_operator_exact_at_gridpoints():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    coords = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]])
    y = ScoreBasedDataAssimilator.observation_operator(x, coords)
    assert y[0, 0, 0] == pytest.approx(0.0)
    assert y[0, 1, 0] == pytest.approx(15.0)


def test_sda_assimilate_runs():
    model = ScoreUNet(base=16)
    assim = ScoreBasedDataAssimilator(model, torch.eye(6))
    ab = cosine_alpha_bars(8)
    x = torch.randn(1, 6, 16, 16)
    y = torch.randn(1, 3, 6)
    coords = torch.zeros(1, 3, 2)
    out = assim.assimilate(x, y, coords, ab, start_frac=0.5)
    assert out.shape == x.shape and torch.isfinite(out).all()


# ---------------- layer 2 ----------------
def test_corrdiff_losses_and_sampling():
    m = CorrDiffCascadedDownscaler(base=16)
    x = torch.randn(2, 6, 32, 32)
    z = torch.randn(2, 2, 32, 32)
    y = torch.randn(2, 6, 32, 32)
    assert m.regression_loss(x, z, y).item() > 0
    assert m.diffusion_loss(x, z, y).item() > 0
    out = m.sample(x, z, n_steps=3)
    assert out.shape == y.shape and torch.isfinite(out).all()


# ---------------- layer 3 ----------------
def test_gnn_score_range_and_rank_invariance():
    g = ContextEnrichedSpatialGNN()
    x = torch.rand(30, 7) * torch.tensor([700, 6, 1, 20, 50, 9, 30])
    coords = torch.rand(30, 2) * 100
    ei, ea = knn_spatial_graph(coords, 5)
    s1 = g(x, ei, ea)
    assert ((0 <= s1) & (s1 <= 1)).all()
    # deterministic on the fixed regional graph -> candidate-pool invariant
    s2 = g(x, ei, ea)
    assert torch.allclose(s1, s2)


# ---------------- layer 4 ----------------
def test_financials_sane():
    fin = project_financials(50.0, 9.9, 4.6, 7.2, 12.0)
    assert fin["capexCr"] > 0 and 0 < fin["lcoeInrKwh"] < 10
    assert fin["p90NetGenerationGwhYr"] < fin["p50NetGenerationGwhYr"]


def test_irr_bisection():
    # -100 now, +20/yr for 10 years -> IRR ~ 15.1%
    assert irr([-100] + [20] * 10) == pytest.approx(0.151, abs=0.01)


def test_workflow_interrupt_and_lossless_resume(tmp_path):
    wf = PersistentParcelWorkflow(SqliteSaver(tmp_path / "ckpt.db"))
    tid = wf.register_parcel({"netBuildableHa": 80, "netBuildablePct": 55,
                              "meanCloudCover": 0.4, "meanWpd120m": 350,
                              "irrPct": 11.0, "suitabilityScore": 0.7})
    st = wf.step(tid, {"currentCloudCover": 0.46, "projectedIrrPct": 8.0})
    assert st["meteoDriftDetected"] and st["hitlPending"]
    # simulated restart
    wf2 = PersistentParcelWorkflow(SqliteSaver(tmp_path / "ckpt.db"))
    assert wf2.saver.get(tid) == st
    resumed = wf2.resume_hitl(tid, "approve")
    assert resumed["stakeholderDecision"] == "approve"
    assert not resumed["hitlPending"]


def test_report_pdf_renders():
    from terraguard.layer4_orchestrator.report_synthesizer import report_to_pdf
    report = {
        "reportId": "t", "generatedAt": "2026-01-01T00:00:00",
        "anchor": {"lat": 14.0, "lon": 76.0}, "inferenceSeconds": 1.0,
        **{f"section{s}": {"title": f"Section {s}", "x": 1}
           for s in "ABCDEF"}}
    assert report_to_pdf(report).startswith(b"%PDF")
