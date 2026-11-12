import React, { useState } from "react";

function Tile({ k, v, u, accent }) {
  return (
    <div className={`tile ${accent ? "accent" : ""}`}>
      <div className="k">{k}</div>
      <div className="v">
        {v}
        {u && <span className="u">{u}</span>}
      </div>
    </div>
  );
}

function KV({ rows }) {
  return (
    <div className="kv">
      {rows.map(([k, v]) => (
        <div key={k}>
          <span>{k}</span>
          <b>{String(v)}</b>
        </div>
      ))}
    </div>
  );
}

function Section({ id, title, open, onToggle, children }) {
  return (
    <div className={`sect ${open ? "open" : ""}`}>
      <button className="sect-head" onClick={onToggle}>
        <span className="badge">{id}</span>
        {title}
        <span className="chev">▸</span>
      </button>
      {open && <div className="sect-body">{children}</div>}
    </div>
  );
}

const CLASS_COLOR = {
  "Urban / Habitation": "#dc2626",
  "Agricultural / Fallow": "#84cc16",
  "Rangeland / Scrubland": "#10b981",
  "Forest / Tree Canopy": "#065f46",
  "Water / Wetland": "#0284c7",
  "Barren / Wasteland": "#f59e0b",
  "Roads / Access Track": "#64748b",
};

export default function ReportViewer({ result, onHitl }) {
  const [open, setOpen] = useState({ A: true, B: true, C: false, D: true, E: false, F: false });
  const [hitlBusy, setHitlBusy] = useState(false);
  const [hitlState, setHitlState] = useState(null);

  if (!result)
    return (
      <div className="card">
        <h2>
          Site Selection Report <span className="tag">6 SECTIONS</span>
        </h2>
        <div className="empty">
          <div className="big">📑</div>
          The investor-grade 6-section report — land audit, micro-climate
          physics, GNN suitability, 25-year financials, drift monitoring and
          HITL governance — appears here after analysis.
        </div>
      </div>
    );

  const r = result.report;
  const { sectionA: A, sectionB: B, sectionC: C, sectionD: D, sectionE: E, sectionF: F } = r;
  const toggle = (k) => setOpen((o) => ({ ...o, [k]: !o[k] }));
  const score = C.suitabilityScore100;
  const circumference = 2 * Math.PI * 62;
  const gradeColor = C.grade === "A" ? "#059669" : C.grade === "B" ? "#d97706" : "#e11d48";

  async function hitl(decision) {
    setHitlBusy(true);
    try {
      const res = await fetch("/api/v1/hitl/resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threadId: r.reportId, decision }),
      });
      const j = await res.json();
      setHitlState(j.state?.stakeholderDecision || null);
      onHitl?.(decision);
    } finally {
      setHitlBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>
        Site Selection Report <span className="tag">{r.reportId}</span>
      </h2>

      <div className="hero" style={{ marginBottom: 16 }}>
        <div className="gauge">
          <svg width="148" height="148" viewBox="0 0 148 148" role="img"
               aria-label={`Suitability score ${score} out of 100`}>
            <circle cx="74" cy="74" r="62" fill="none" stroke="rgba(255,255,255,.07)" strokeWidth="11" />
            <circle
              cx="74" cy="74" r="62" fill="none"
              stroke={gradeColor} strokeWidth="11" strokeLinecap="round"
              strokeDasharray={`${(score / 100) * circumference} ${circumference}`}
              style={{ transition: "stroke-dasharray 1.2s cubic-bezier(.2,.8,.2,1)" }}
            />
          </svg>
          <div className="val">
            <div style={{ textAlign: "center" }}>
              <b>{score}</b>
              <div><span>SUITABILITY</span></div>
            </div>
          </div>
        </div>
        <div className="hero-meta">
          <span className={`grade-chip grade-${C.grade}`}>GRADE {C.grade}</span>
          <div className="tech">{C.recommendedTechnology}</div>
          <div className="sub">
            {C.regionalPercentile}th regional percentile · rank-invariant fuzzy
            GNN score · inference {r.inferenceSeconds}s
          </div>
          <div className="toolrow">
            <a className="btn ghost" href={`/api/v1/report/${r.reportId}/pdf`}>
              ⬇ PDF report
            </a>
          </div>
        </div>
      </div>

      <Section id="A" title={A.title} open={open.A} onToggle={() => toggle("A")}>
        <div className="tiles" style={{ marginBottom: 12 }}>
          <Tile k="Total footprint" v={A.totalFootprintHa} u="ha" />
          <Tile k="Net buildable" v={A.netBuildableHa} u="ha" accent />
          <Tile k="Buildable" v={A.netBuildablePct} u="%" />
          <Tile k="Solar capacity" v={A.plannedSolarMwp} u="MWp" />
          <Tile k="Turbines" v={A.plannedTurbines} u={`× 3.3 MW`} />
        </div>
        <div className="bars">
          {Object.entries(A.classDistributionPct)
            .sort((a, b) => b[1] - a[1])
            .map(([nm, pct]) => (
              <div className="bar-row" key={nm} title={`${nm}: ${pct}%`}>
                <span className="nm">{nm}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{ width: `${pct}%`, background: CLASS_COLOR[nm] || "#64748b" }}
                  />
                </div>
                <span className="pv">{pct}%</span>
              </div>
            ))}
        </div>
        <div style={{ height: 10 }} />
        <KV
          rows={[
            ["Mean roughness z₀", `${A.meanRoughnessZ0} m`],
            ["Surface albedo α", A.meanSurfaceAlbedo],
            ["Internal road access", A.internalRoadAccess ? "Yes" : "No"],
            ["Setbacks applied", Object.entries(A.setbacksAppliedM).map(([k, v]) => `${k.split(" ")[0]} ${v}m`).join(" · ")],
          ]}
        />
      </Section>

      <Section id="B" title={B.title} open={open.B} onToggle={() => toggle("B")}>
        <div className="tiles" style={{ marginBottom: 12 }}>
          <Tile k="Mean wind @120m" v={B.niweMeanWind120m} u="m/s" accent />
          <Tile k="PINN WPD" v={B.pinnWpd120mWm2} u="W/m²" />
          <Tile k="G_eff solar" v={B.geffKwhM2Day} u="kWh/m²/d" accent />
          <Tile k="Ridge speed-up" v={`+${B.ridgelineSpeedupPct}`} u="%" />
        </div>
        <KV
          rows={[
            ["Weibull A / k", `${B.weibull.A} / ${B.weibull.k}`],
            ["Air density ρ", `${B.airDensityKgM3} kg/m³`],
            ["Peak WPD", `${B.pinnMaxWpdWm2} W/m²`],
            ["GHI (clear-sky)", `${B.ghiKwhM2Day} kWh/m²/d`],
            ["Mean cloud cover", B.meanCloudCover],
            ["PM2.5 / PM10", `${B.pm25UgM3} / ${B.pm10UgM3} µg/m³`],
            ["Soiling attenuation", `${B.annualSoilingAttenuationPct}% /yr`],
            ["Terrain slope", `${B.terrainSlopeDeg}°`],
            ["SDA RMSE reduction", `${B.sdaCalibration.rmseReductionPct}% (${B.sdaCalibration.mastStations} masts)`],
            ["Bias correction", `×${B.sdaCalibration.biasCorrectionFactor}`],
          ]}
        />
      </Section>

      <Section id="C" title={C.title} open={open.C} onToggle={() => toggle("C")}>
        <KV
          rows={[
            ["Suitability score", `${C.suitabilityScore100} / 100 (Grade ${C.grade})`],
            ["Technology", C.recommendedTechnology],
            ["Wind membership μ", C.windMembership],
            ["Solar membership μ", C.solarMembership],
            ["Nearest ≥66 kV substation", `${C.nearestSubstation.name} (${C.nearestSubstation.voltageKv} kV) · ${C.nearestSubstationKm} km`],
            ["Nearest highway", `${C.nearestHighway?.name ?? "—"} · ${C.nearestHighwayKm} km`],
            ["Parcel permissible", C.parcelPermissible ? "Yes" : "No"],
            ["Exclusion buffers", Object.entries(C.exclusionBuffersM).map(([k, v]) => `${k} ${v}m`).join(" · ")],
          ]}
        />
      </Section>

      <Section id="D" title={D.title} open={open.D} onToggle={() => toggle("D")}>
        <div className="tiles" style={{ marginBottom: 12 }}>
          <Tile k="IRR (25-yr)" v={D.irrPct} u="%" accent />
          <Tile k="NPV" v={`₹${D.npvCr}`} u="Cr" />
          <Tile k="LCOE" v={`₹${D.lcoeInrKwh}`} u="/kWh" accent />
          <Tile k="CAPEX" v={`₹${D.capexCr}`} u="Cr" />
          <Tile k="P50 energy" v={D.p50NetGenerationGwhYr} u="GWh/yr" />
          <Tile k="P90 energy" v={D.p90NetGenerationGwhYr} u="GWh/yr" />
        </div>
        <KV
          rows={[
            ["Installed solar", `${D.installedSolarMwp} MWp (CUF ${D.cufSolarPct}%)`],
            ["Installed wind", `${D.installedWindMw} MW (CUF ${D.cufWindPct}%)`],
            ["PPA tariff", `₹${D.ppaTariffInrKwh}/kWh`],
            ["Evacuation line", `${D.txLineKm} km · ₹${D.txLineCostCr} Cr`],
            ["O&M", `₹${D.omCrYr} Cr/yr`],
            ["Discount rate", `${D.discountRatePct}%`],
          ]}
        />
      </Section>

      <Section id="E" title={E.title} open={open.E} onToggle={() => toggle("E")}>
        <KV
          rows={[
            ["Monitoring thread", E.monitoringThreadId],
            ["Checkpoint backend", E.checkpointBackend],
            ["Cloud-drift trigger", `≥${E.cloudDriftTriggerPct}%`],
            ["WPD-drift trigger", `≥${E.wpdDriftTriggerPct}%`],
            ["ΔNDVI / ΔNDBI triggers", `${E.ndviDriftTrigger} / ${E.ndbiDriftTrigger}`],
            ["Sentinel-2 revisit", `${E.revisitCadenceDays} days`],
            ["10-yr cloud margin", `${E.tenYearCloudDriftMarginPct}%`],
            ["10-yr WPD margin", `${E.tenYearWpdDriftMarginPct}%`],
          ]}
        />
      </Section>

      <Section id="F" title={F.title} open={open.F} onToggle={() => toggle("F")}>
        <p style={{ fontSize: 12.5, color: "var(--ink-2)", marginTop: 0 }}>
          If projected IRR falls below the {F.irrFloorPct}% investor floor, the
          LangGraph workflow raises <code>interrupt()</code> and freezes on its
          durable checkpoint until a stakeholder signs off here.
        </p>
        <div className="hitl">
          <button className="btn" disabled={hitlBusy} onClick={() => hitl("approve")}>
            ✓ Approve site
          </button>
          <button className="btn danger" disabled={hitlBusy} onClick={() => hitl("abandon")}>
            ✕ Abandon site
          </button>
          <span className="state">
            {hitlState ? `Recorded: ${hitlState.toUpperCase()}` : `Status: ${F.status}`}
          </span>
        </div>
      </Section>
    </div>
  );
}
