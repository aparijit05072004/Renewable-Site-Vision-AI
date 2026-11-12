# TerraGuard-v2 — Validation Report

Consolidated results of the blueprint's quantitative acceptance gates
(Table 4 of the TerraGuard-v2 engineering specification). Every number
below was measured by `python -m terraguard.training.train_all` on CPU;
raw JSON lives in `artifacts/metrics/`.

**Overall: ALL GATES PASSED ✅** (`validation_suite.json: allGatesPassed = True`)

## Phase 1 — Layer 0 Vision Segmentation Gate
| Metric | Gate | Measured | Result |
|---|---|---|---|
| mIoU (Barren/Scrub/Water/Forest, held-out) | ≥ 0.84 | **0.8721** | ✅ |
| Inference latency per 1024×1024 image | < 1.5 s | **0.056 s** | ✅ |

Per-class IoU: {"0": 0.6945, "1": 0.811, "2": 0.8233, "3": 0.9198, "4": 0.8389, "5": 0.9063, "6": 0.4704} · 998,615 params · 700 steps (3.5 min)

## Phase 2 — Layer 1 SDA Accuracy Gate
| Metric | Gate | Measured | Result |
|---|---|---|---|
| Held-out mast wind RMSE reduction vs bilinear prior | ≥ 25 % | **30.9 %** | ✅ |

Per-trial: [31.2, 51.9, 48.5, 11.5, 27.1, 15.1] · 48 guided masts, 12 held-out, 4-member posterior ensemble, guidance 12.0, start-frac 0.5, 64 diffusion steps

## Phase 3 — Layer 2 Physics & Spectral Gate
| Metric | Gate | Measured | Result |
|---|---|---|---|
| CorrDiff energy-spectrum slope vs truth (k^-5/3) | within ±0.45 & shallower than MSE-only | truth **-1.619**, CorrDiff **-1.205**, MSE-only -2.17 | ✅ |
| PINN continuity residual ‖R_mass‖₂ | < 1e-4 | **4.64e-08** (divergence-free by construction) | ✅ |

CorrDiff: ×8 downscale to 128² grid, normalized u-RMSE 0.6462. PINN: momentum-x residual RMS 0.178 (soft-constrained), 41,860 params.

## Phase 4 — Layer 3 Rank Stability Gate
| Metric | Gate | Measured | Result |
|---|---|---|---|
| Kendall τ, surviving top-25 after deleting 15 % of candidates | = 1.00 | **1.0** (all trials [1.0, 1.0, 1.0, 1.0, 1.0]) | ✅ |
| Kendall τ vs ground-truth project yields | ≥ 0.8 | **0.9391** | ✅ |

Rank Reversal is eliminated structurally: scores are computed on the fixed
regional lattice with domain-anchored fuzzy memberships and are never
normalised over the candidate pool.

## Phase 5 — End-to-End & Persistence Gate
| Metric | Gate | Measured | Result |
|---|---|---|---|
| Raw land image → full 6-section report | < 6 s | **1.8 s** | ✅ |
| +6 % cloud-drift triggers HITL interrupt | yes | **True** | ✅ |
| Lossless state resumption after process restart | yes | **True** (HITL resume ok: True) | ✅ |

Report sections generated: sectionA, sectionB, sectionC, sectionD, sectionE, sectionF · PDF: 5221 bytes.

## Test suite
`python -m pytest tests/` — 12/12 passing (field spectra, geo resolver,
Layer-0 analysis, SDA operator & assimilation, CorrDiff losses/sampler,
GNN score bounds & determinism, financial IRR/LCOE, workflow interrupt +
lossless resume, PDF rendering).

## Scope note
Models are trained on spectrum-matched procedural data (the open-data
extraction protocols in `terraguard/data_acquisition/` are shape-compatible
drop-ins), so gate numbers certify the *architecture, training recipes and
system behaviour*, not real-world siting accuracy. Production deployment
swaps in DeepGlobe/LoveDA, ARCO-ERA5, NIWE/GWA, NSRDB/IMDAA, GLO-30 and GEM
feeds per the blueprint without code changes to the models.
