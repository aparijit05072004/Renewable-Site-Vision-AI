# TerraGuard-v2 — Renewable Site Vision AI 🌍⚡

**Multi-Agent Micro-Climate Forecasting & Image-to-Report Spatial Suitability Engine**

Upload any land parcel image — a georeferenced GeoTIFF, a GPS-tagged drone
photo, or a plain aerial screenshot with a map pin — and TerraGuard-v2 runs a
five-layer AI cascade that returns an investor-grade **Renewable Energy Site
Selection Report** with an annotated solar/wind layout blueprint drawn over
your image, plus a persistent multi-agent monitoring thread.

```
 land image ─► L0 Vision ─► L1 SDA ─► L2 Macro→CorrDiff→PINN ─► L3 GNN ─► L4 Report+Agents
              SegFormer     guided     GraphCast/Aurora        fuzzy       6 sections,
              7-class       reverse    surrogate · 2 km EDM    GATv2       PDF, HITL,
              masks, z0, α  diffusion  diffusion · ABL PDEs    score       drift agents
```

## Architecture (per the TerraGuard-v2 blueprint)

| Layer | Module | What it does |
|---|---|---|
| **0 · Vision** | `terraguard/layer0_vision/` | GeoTIFF/EXIF/pin geolocation resolver; SegFormer-style MiT transformer segmenting 7 siting classes; Euclidean setback dilation; z₀(x,y) & albedo grids; buildable-area & capacity packing equations; PV-row + 7D-spaced turbine layout renderer |
| **1 · SDA** | `terraguard/layer1_sda/` | Denoising score U-Net + guided reverse-time diffusion fusing sparse 120 m mast observations into coarse atmospheric states (Bayes-in-score-space, Tweedie denoising, DDIM) |
| **2 · Atmosphere** | `terraguard/layer2_atmosphere/` | GraphCast/Aurora macro stage (earth2studio adapter + CPU spectral surrogate); **CorrDiff** two-step downscaler (regression U-Net mean + EDM diffusion residual, Karras-Heun sampler) 25 km→2 km; **ABL-PINN** enforcing continuity + Navier-Stokes/Ekman PDEs with image-derived z₀ no-slip boundary; WPD & aerosol-soiled bifacial G_eff physics |
| **3 · Spatial** | `terraguard/layer3_spatial/` | PostGIS Boolean exclusion SQL (+ dependency-free fallback engine); **Fuzzy Context-Enriched Spatial GNN** — domain-anchored learnable sigmoid memberships + self-contained GATv2 attention + pairwise rank loss ⇒ provably Rank-Reversal-free scores |
| **4 · Agents** | `terraguard/layer4_orchestrator/` | Cyclic multi-agent workflow (meteo-drift / geo-regulatory / financial / HITL-gate nodes) with durable checkpointing (SQLite saver, PostgresSaver-compatible; LangGraph wiring included); 6-section report + PDF; 25-yr financials (CUF, P50/P90, CAPEX, LCOE, NPV, IRR); FastAPI service; **TERRA** AI assistant |

## Trained models & validation gates (all run on CPU, checkpoints in `artifacts/checkpoints/`)

Every gate from the blueprint's execution roadmap (Table 4) is executed for
real by `python -m terraguard.training.train_all`; consolidated results land
in `artifacts/metrics/validation_suite.json`:

1. **Vision gate** — mIoU on Barren/Scrub/Water/Forest + <1.5 s/1024² latency
2. **SDA gate** — ≥25 % RMSE reduction at held-out masts vs bilinear interpolation
3. **Physics & spectral gate** — CorrDiff restores the k^−5/3 spectrum; PINN continuity residual
4. **Rank-stability gate** — Kendall τ = 1.00 on top-25 after deleting 15 % of candidates
5. **E2E & persistence gate** — image→6-section report < 6 s; +6 % cloud-drift interrupt; lossless resume after restart

Training data: spectrum-matched procedural generators (`terraguard/common/fields.py`,
`layer0_vision/synthetic_tiles.py`) stand in for the open datasets named in the
blueprint (DeepGlobe/LoveDA, ARCO-ERA5, NIWE/GWA, NSRDB/IMDAA, Copernicus DEM,
GEM trackers) so the full stack trains anywhere; the real extraction protocols
ship in `terraguard/data_acquisition/` and are shape-compatible drop-ins.

## Quickstart

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m terraguard.training.train_all        # Phases 1-5: train + gates (~30 min CPU)
uvicorn terraguard.layer4_orchestrator.api_server:app --port 8000
```

Dashboard (dev):

```bash
cd dashboard-ui && npm install && npm run dev   # http://localhost:5173 (proxies /api → :8000)
```

Or build once (`npm run build`) and the FastAPI server serves it at `http://localhost:8000`.

Full stack with PostgreSQL 16 + PostGIS 3.4:

```bash
docker compose up --build
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/analyze-land-image` | multipart image (+ optional `pinLat`,`pinLon`,`gsdM`) → report JSON + overlays + heatmaps |
| `GET /api/v1/report/{id}` / `.../pdf` | stored report / downloadable PDF |
| `POST /api/v1/monitor/step` | advance a monitoring cycle (drift agents) |
| `POST /api/v1/hitl/resume` | stakeholder sign-off (`approve`/`abandon`) |
| `POST /api/v1/assistant/chat` | TERRA assistant (report-grounded; optional Claude tier via `ANTHROPIC_API_KEY`) |
| `GET /api/v1/threads` | persistent monitoring threads |

## TERRA — the site-intelligence assistant 🤖

The dashboard ships a floating neon assistant orb. TERRA answers from the
**live report JSON** (site summary, score rationale, wind-vs-solar case,
financials, drift status, grid logistics) and from a built-in domain knowledge
base (WPD, Weibull, CorrDiff, PINN, SDA, GNN/rank-reversal, LCOE, CUF…). With
`ANTHROPIC_API_KEY` set it escalates free-form questions to Claude with the
report as grounding context; without it, everything works fully offline.

## Repository layout

```
terraguard/            Python engine (layers 0-4, training, pipeline, API)
dashboard-ui/          React + Vite dashboard with TERRA assistant
artifacts/             trained checkpoints + gate metrics + figures
tests/                 pytest suite
docs/                  model cards & validation notes
docker-compose.yml     PostGIS 16/3.4 + API + UI stack
```

## Documentation

- `docs/MODEL_CARDS.md` — architecture, data, training & measured metrics per model
- `docs/VALIDATION.md` — gate-by-gate results of the latest run
