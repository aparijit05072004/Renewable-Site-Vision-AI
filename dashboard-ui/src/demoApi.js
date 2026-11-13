// Static demo mode for GitHub Pages: serves the real trained pipeline's
// precomputed output (public/demo/analysis.json) and runs a JS port of
// TERRA's grounded intent engine, since Pages cannot host the PyTorch API.
export const DEMO = import.meta.env.VITE_DEMO === "1";

const BASE = import.meta.env.BASE_URL || "/";
let _payload = null;

export async function demoAnalyze() {
  if (!_payload) {
    const res = await fetch(`${BASE}demo/analysis.json`);
    if (!res.ok) throw new Error("demo bundle missing");
    _payload = await res.json();
  }
  return {
    ..._payload,
    originalUrl: `${BASE}demo/sample-parcel.png`,
    fileName: "sample-parcel.png (demo)",
  };
}

const KNOWLEDGE = {
  corrdiff:
    "CorrDiff is NVIDIA's generative downscaling recipe used in Layer 2: a " +
    "regression U-Net predicts the conditional mean of the 2 km fields from " +
    "the 25 km state + terrain, then an EDM score-matching diffusion model " +
    "generates the stochastic residual. That restores the k^-5/3 turbulence " +
    "spectrum an MSE-only model smooths away — here the sampled slope is " +
    "-1.21 vs -2.17 for pure regression.",
  pinn:
    "The ABL-PINN is a physics-informed network for hub-height micro-siting. " +
    "It parametrizes wind as the curl of a learned vector potential, so mass " +
    "continuity (∇·u = 0) holds exactly by construction (~1e-6 residual), " +
    "while Navier-Stokes/Ekman momentum residuals and a log-law wall " +
    "boundary shape the 120 m flow over the terrain.",
  sda:
    "Score-Based Data Assimilation (Layer 1) treats a diffusion model over " +
    "atmospheric states as a learned prior, then guides the reverse-time " +
    "sampler with mast observations (Bayes' rule in score space). The " +
    "posterior ensemble mean cut held-out mast RMSE by 30.9%.",
  weibull:
    "Wind speeds follow a Weibull distribution with scale A and shape k. " +
    "TerraGuard fits them from the PINN-corrected 120 m mean speed; they " +
    "feed the P50/P90 energy estimates in Section D.",
  wpd:
    "Wind Power Density = ½·ρ·v³ (W/m²) — the kinetic energy flux through " +
    "the rotor plane at 120 m, using CorrDiff-downscaled wind and the " +
    "macro-stage air density.",
  lcoe:
    "LCOE = discounted lifetime cost ÷ discounted lifetime energy (INR/kWh)." +
    " TerraGuard discounts 25 years of O&M plus CAPEX against degraded P50 " +
    "generation at a 10.5% rate.",
  cuf:
    "CUF (capacity-utilisation factor) is actual annual energy ÷ nameplate " +
    "× 8760 h. Solar CUF comes from effective irradiance with a tracker " +
    "performance ratio; wind CUF from a simplified power-curve integral of " +
    "the 120 m mean speed.",
  "rank reversal":
    "Rank reversal is when removing one candidate reorders the others — " +
    "fatal for land-ranking tools. TerraGuard's GNN scores every parcel on " +
    "a fixed regional lattice (never normalised to the candidate pool), so " +
    "deleting 15% of candidates leaves Kendall τ = 1.00.",
  segformer:
    "Layer 0 uses a SegFormer-style hierarchical Mix-Transformer encoder " +
    "with an all-MLP decoder to label every pixel into 7 land-cover classes" +
    " (mIoU 0.872), from which buildable masks, roughness z₀ and albedo " +
    "grids are derived.",
  gnn:
    "Layer 3's suitability model is a GATv2 graph-attention network over a " +
    "k-NN spatial graph, fed by learnable fuzzy memberships whose sigmoid " +
    "inflections sit at physical thresholds (e.g. 200 W/m² WPD).",
};

function fmt(n, d = 1) {
  return typeof n === "number" ? n.toFixed(d) : String(n);
}

export function demoChat(message) {
  const q = message.toLowerCase();
  const r = _payload?.report;
  for (const [k, v] of Object.entries(KNOWLEDGE)) {
    if (q.includes(k)) return v;
  }
  if (!r)
    return (
      "Run the demo analysis first (hit the dropzone) and I'll ground my " +
      "answers in the live report. Meanwhile, ask me about CorrDiff, PINN, " +
      "SDA, Weibull, WPD, LCOE, CUF or rank reversal."
    );
  const A = r.sectionA, B = r.sectionB, C = r.sectionC, D = r.sectionD,
        E = r.sectionE;
  if (q.includes("summar") || q.includes("overview"))
    return (
      `Site verdict: Grade ${C.grade} (${fmt(C.suitabilityScore100)}/100), ` +
      `recommended build: ${C.recommendedTechnology}.\n\n` +
      `• ${fmt(A.netBuildableHa)} ha buildable (${fmt(A.netBuildablePct)}% of parcel) → ` +
      `${fmt(A.plannedSolarMwp)} MWp solar + ${A.plannedTurbines} turbines\n` +
      `• 120 m wind ${fmt(B.niweMeanWind120m, 2)} m/s · WPD ${fmt(B.pinnWpd120mWm2)} W/m² · ` +
      `G_eff ${fmt(B.geffKwhM2Day, 2)} kWh/m²/day\n` +
      `• P50 ${fmt(D.p50NetGenerationGwhYr)} GWh/yr · IRR ${fmt(D.irrPct, 2)}% · ` +
      `LCOE ₹${fmt(D.lcoeInrKwh, 2)}/kWh · NPV ₹${fmt(D.npvCr)} Cr\n` +
      `• Grid: substation ${fmt(C.nearestSubstationKm)} km · ` +
      `permissible: ${C.parcelPermissible ? "yes" : "no"}`
    );
  if (q.includes("wind") || q.includes("solar"))
    return (
      `Fuzzy memberships — wind ${fmt(C.windMembership, 2)} vs solar ` +
      `${fmt(C.solarMembership, 2)} → ${C.recommendedTechnology}.\n\n` +
      `Wind case: ${fmt(B.niweMeanWind120m, 2)} m/s at 120 m (Weibull A=` +
      `${fmt(B.weibull.A, 2)}, k=${fmt(B.weibull.k, 2)}), WPD ` +
      `${fmt(B.pinnWpd120mWm2)} W/m², ridgeline speed-up +` +
      `${fmt(B.ridgelineSpeedupPct)}%.\nSolar case: G_eff ` +
      `${fmt(B.geffKwhM2Day, 2)} kWh/m²/day after ` +
      `${fmt(B.annualSoilingAttenuationPct, 1)}% aerosol soiling, solar CUF ` +
      `${fmt(D.cufSolarPct)}%.`
    );
  if (q.includes("financ") || q.includes("irr") || q.includes("npv") ||
      q.includes("capex"))
    return (
      `25-year financials for ${fmt(D.installedSolarMwp)} MWp solar + ` +
      `${fmt(D.installedWindMw)} MW wind:\n\n` +
      `• CAPEX ₹${fmt(D.capexCr)} Cr (incl. ${fmt(D.txLineKm)} km tx line)\n` +
      `• P50 ${fmt(D.p50NetGenerationGwhYr)} / P90 ${fmt(D.p90NetGenerationGwhYr)} GWh/yr ` +
      `at ${fmt(D.cufCombinedPct)}% combined CUF\n` +
      `• LCOE ₹${fmt(D.lcoeInrKwh, 2)}/kWh vs PPA ₹${fmt(D.ppaTariffInrKwh, 2)}/kWh\n` +
      `• NPV ₹${fmt(D.npvCr)} Cr @ ${fmt(D.discountRatePct)}% · IRR ${fmt(D.irrPct, 2)}%` +
      ` (HITL floor 9.5%)`
    );
  if (q.includes("score") || q.includes("why") || q.includes("grade"))
    return (
      `The GATv2 GNN scored this parcel ${fmt(C.suitabilityScore100)}/100 ` +
      `(Grade ${C.grade}, ${fmt(C.regionalPercentile)}th regional percentile). ` +
      `Drivers: WPD ${fmt(B.pinnWpd120mWm2)} W/m², buildable ratio ` +
      `${fmt(A.netBuildablePct)}%, slope ${fmt(B.terrainSlopeDeg, 1)}°, ` +
      `substation ${fmt(C.nearestSubstationKm)} km, soiling ` +
      `${fmt(B.annualSoilingAttenuationPct, 1)}%. Scores live on a fixed ` +
      `regional lattice, so they're immune to rank reversal (τ = 1.00).`
    );
  if (q.includes("monitor") || q.includes("drift") || q.includes("status"))
    return (
      `Monitoring thread ${E.monitoringThreadId} is ARMED on a ` +
      `${E.revisitCadenceDays}-day revisit cadence. Interrupt triggers: cloud ` +
      `drift >${fmt(E.cloudDriftTriggerPct)}%, WPD drift >` +
      `${fmt(E.wpdDriftTriggerPct)}%, NDVI ${E.ndviDriftTrigger}, NDBI ` +
      `${E.ndbiDriftTrigger}. In this static demo the cyclic agents don't ` +
      `run — clone the repo and \`docker compose up\` for live monitoring + ` +
      `HITL sign-off.`
    );
  if (q.includes("land") || q.includes("parcel") || q.includes("buildable"))
    return (
      `Layer-0 audit: ${fmt(A.totalFootprintHa)} ha footprint, ` +
      `${fmt(A.netBuildableHa)} ha net buildable after setbacks. Mean ` +
      `roughness z₀ ${A.meanRoughnessZ0} m, albedo ${A.meanSurfaceAlbedo}. ` +
      `Packing: ${fmt(A.plannedSolarMwp)} MWp PV + ${A.plannedTurbines} × 4.5 MW turbines.`
    );
  return (
    "I'm the static-demo TERRA, grounded in this parcel's precomputed " +
    "report. Try: 'summarize this site', 'wind or solar?', 'explain the " +
    "financials', 'why this score?', 'monitoring status' — or ask about " +
    "CorrDiff, PINN, SDA, SegFormer, GNN, Weibull, WPD, LCOE, CUF, rank " +
    "reversal. The full Claude-powered TERRA runs with the local API server."
  );
}
