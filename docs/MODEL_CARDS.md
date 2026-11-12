# TerraGuard-v2 — Model Cards

All five trainable models in the cascade are trained on CPU by
`python -m terraguard.training.train_all`. Checkpoints land in
`artifacts/checkpoints/`, per-model gate metrics in `artifacts/metrics/`.
Training data comes from spectrum-matched procedural generators
(`terraguard/common/fields.py`, `terraguard/layer0_vision/synthetic_tiles.py`)
that stand in for the open datasets named in the blueprint; the real
extraction protocols (`terraguard/data_acquisition/`) produce tensors of the
same shapes and are drop-in replacements for production GPU runs.

---

## 1 · Layer 0 — SegFormer-Nano land segmentation

| | |
|---|---|
| Architecture | SegFormer (Xie et al.): 4-stage Mix-Transformer encoder — overlapping patch embeds, spatial-reduction efficient self-attention, Mix-FFN with depthwise conv — + all-MLP decoder. "Nano" dims (24/48/96/160), ~1.0 M params |
| Task | 7-class renewable-siting land cover (urban, agri, scrub, forest, water, barren, roads) |
| Training | ~700 steps AdamW, class-weighted cross-entropy, cosine LR, procedural DeepGlobe/LoveDA-morphology tiles (192²) |
| Production path | swap in `nvidia/segformer-b2-finetuned-ade-512-512` via HuggingFace, fine-tune on DeepGlobe + LoveDA Rural + Dynamic World India (`GOOGLE/DYNAMICWORLD/V1`) |
| Gate | mIoU ≥ 0.84 on Barren/Scrub/Water/Forest held-out tiles; < 1.5 s per 1024² image |

## 2 · Layer 1 — Score U-Net + guided reverse-time SDA

| | |
|---|---|
| Architecture | Time-conditioned residual U-Net (~1.9 M params), epsilon-prediction DDPM, 64-step cosine schedule, over [6, 32, 32] atmospheric states (u10, v10, t2m, sp, tcc, q) |
| Assimilation | Bayes in score space: Tweedie denoised estimate → bilinear observation operator at mast sites → precision-weighted likelihood gradient added to the score; DDIM integration from an SDEdit-style partially-diffused prior; posterior = 4-member guided-ensemble mean |
| Tuned settings | 48 guided masts, guidance scale 12.0, start fraction 0.5, obs precision 6·I |
| Production path | NIWE 120 m Wind Atlas + Global Wind Atlas v4.0 COGs + NOAA ISD masts guiding ARCO-ERA5 states (earth2studio SDA workflow) |
| Gate | ≥ 25 % wind-component RMSE reduction at 12 held-out masts vs bilinear interpolation of the biased coarse prior |

## 3 · Layer 2B — CorrDiff cascaded downscaler (25 km → 2 km)

| | |
|---|---|
| Architecture | NVIDIA CorrDiff two-step: (1) regression U-Net μ₀(x_coarse↑, z_topo) trained with MSE; (2) EDM score-matching diffusion U-Net Dφ on the residual r = y − μ₀ (Karras preconditioning c_skip/c_out/c_in, log-normal σ sampling), 18/14-step Heun sampler. ~1.5 M params total, 16 → 128 grid (×8) |
| Training pairs | coarse synoptic states (k⁻³) vs fine truth = bilinear↑ + terrain forcing (lapse rate, orographic cloud, upslope speed-up) + k^−5/3 stochastic turbulence |
| Production path | paired ARCO-ERA5 25 km ↔ NREL India NSRDB / NCMRWF IMDAA 2–12 km reanalysis tensors |
| Gate | radially-averaged kinetic-energy spectrum of generated u-fields within ±0.45 log-log slope of truth (≈ −5/3) **and** strictly shallower than the MSE-only regression output (which over-smooths by design) |

## 4 · Layer 2C — Divergence-free ABL-PINN

| | |
|---|---|
| Architecture | Fourier-feature MLP (5×96 tanh, ~42 k params) emitting a vector potential A(x,y,z) + pressure; velocity u = ∇×A, so incompressibility holds **identically** (hard-constrained PINN) |
| Physics losses | steady Navier-Stokes/Ekman x & y momentum residuals (advection, Coriolis vs geostrophic wind, pressure gradient, eddy viscosity), no-slip at z = h_DEM + z₀(x,y) from the Layer-0 image, geostrophic recovery at the ABL top, log-law anchor at 120 m hub height |
| Production path | 30 m Copernicus GLO-30 DEM + 10 m ESA WorldCover z₀ via keyless STAC |
| Gate | ‖R_mass‖₂ < 10⁻⁴ over 4096 collocation points (satisfied by construction; measured each run) |

## 5 · Layer 3B — Fuzzy Context-Enriched Spatial GNN

| | |
|---|---|
| Architecture | Learnable sigmoid fuzzy memberships with *physical* inflection centres (WPD 320 W/m², G_eff 4.4 kWh/m²/day, buildable 0.45, slope 8°, substation 18 km, soiling 4 %, highway 12 km) → 2× GATv2 attention (self-contained Brody-et-al implementation, 4+2 heads, 3-dim edge attrs) → sigmoid scoring head. ~90 k params |
| Loss | pairwise margin ranking + MSE calibration against project-outcome yields (GEM Operating vs Shelved/Cancelled analogue) |
| Rank-reversal immunity | scores are computed on the **fixed regional H3-style lattice**, never normalised over the candidate pool — adding/removing candidates cannot perturb any other site's score (the AHP failure mode) |
| Production path | Global Energy Monitor India Solar/Wind tracker labels on Uber H3 res-7 hexagons |
| Gates | Kendall τ = 1.00 on the surviving top-25 after deleting 15 % of low-ranked candidates; τ ≥ 0.8 vs ground-truth yields |

---

## Non-trained components

- **Macro stage (Layer 2A):** `SynopticSurrogateEngine` — AR(1) spectral rollout
  with monsoonal mean flow, diurnal cycle, Aurora-style dynamic air density
  ρ = p/(R_d·T_v) and PM2.5/PM10 advection; `Earth2StudioAdapter` is the
  integration point for real GraphCast/Aurora checkpoints on GPU hosts.
- **Layer 3A exclusions:** PostGIS 16/3.4 SQL (`postgis_clipping.sql`) with a
  dependency-free haversine fallback engine.
- **Layer 4 agents:** deterministic drift/regulatory/financial/HITL nodes over
  a durable checkpointer (SQLite with the PostgresSaver interface; LangGraph
  wiring provided in `build_langgraph_workflow`).
