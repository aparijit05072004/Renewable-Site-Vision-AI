import React, { useEffect, useRef, useState } from "react";

/* Sequential single-hue ramps (light -> dark reads low -> high on the dark
   surface we invert: low opacity -> saturated). One hue per measure. */
const RAMPS = {
  wpd: (t) => `rgba(2, 132, 199, ${0.08 + 0.85 * t})`, // blue = wind power
  geff: (t) => `rgba(217, 119, 6, ${0.08 + 0.85 * t})`, // amber = solar
  dem: (t) => `rgba(139, 92, 246, ${0.08 + 0.85 * t})`, // violet = elevation
};
const LABELS = {
  original: "Uploaded image",
  segmentation: "7-class land cover",
  layout: "Solar/wind layout blueprint",
  wpd: "Wind power density @120 m (W/m²)",
  geff: "Effective solar irradiance (kWh/m²/day)",
  dem: "Terrain elevation (m)",
};

export default function MapViewport({ result }) {
  const [view, setView] = useState("layout");
  const canvasRef = useRef(null);
  const [hover, setHover] = useState(null);

  const heat = result?.heatmap?.[view];
  useEffect(() => {
    if (!heat || !canvasRef.current) return;
    const n = heat.length;
    const cv = canvasRef.current;
    const px = 10;
    cv.width = n * px;
    cv.height = n * px;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#0e1628";
    ctx.fillRect(0, 0, cv.width, cv.height);
    let mn = Infinity,
      mx = -Infinity;
    heat.forEach((r) => r.forEach((v) => { mn = Math.min(mn, v); mx = Math.max(mx, v); }));
    const ramp = RAMPS[view];
    heat.forEach((row, j) =>
      row.forEach((v, i) => {
        const t = (v - mn) / (mx - mn + 1e-9);
        ctx.fillStyle = ramp(t);
        ctx.fillRect(i * px + 0.5, j * px + 0.5, px - 1, px - 1);
      })
    );
  }, [heat, view]);

  if (!result)
    return (
      <div className="card">
        <h2>
          Spatial Viewport <span className="tag">WEBGL · COG</span>
        </h2>
        <div className="empty">
          <div className="big">🗺️</div>
          Upload a land parcel image to render the segmentation overlay, the
          automated solar/wind layout blueprint, and the 2&nbsp;km
          micro-climate heatmaps here.
        </div>
      </div>
    );

  const { overlays } = result;
  const isImg = ["original", "segmentation", "layout"].includes(view);
  const imgSrc =
    view === "original"
      ? result.originalUrl
      : view === "segmentation"
        ? `data:image/png;base64,${overlays.segmentationPng}`
        : `data:image/png;base64,${overlays.layoutPng}`;

  const stats = heat
    ? (() => {
        let mn = Infinity, mx = -Infinity, sum = 0, c = 0;
        heat.forEach((r) => r.forEach((v) => { mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; c++; }));
        return { mn, mx, mean: sum / c };
      })()
    : null;

  return (
    <div className="card">
      <h2>
        Spatial Viewport <span className="tag">PARCEL INSPECTOR</span>
      </h2>
      <div className="view-tabs" role="tablist">
        {Object.keys(LABELS).map((k) => (
          <button key={k} className={view === k ? "on" : ""} onClick={() => setView(k)}>
            {k.toUpperCase()}
          </button>
        ))}
      </div>
      <div className="viewport">
        {isImg ? (
          <img src={imgSrc} alt={LABELS[view]} />
        ) : (
          <canvas
            ref={canvasRef}
            style={{ imageRendering: "pixelated" }}
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const n = heat.length;
              const i = Math.min(n - 1, Math.floor(((e.clientX - rect.left) / rect.width) * n));
              const j = Math.min(n - 1, Math.floor(((e.clientY - rect.top) / rect.height) * n));
              setHover({ i, j, v: heat[j][i] });
            }}
            onMouseLeave={() => setHover(null)}
          />
        )}
      </div>
      {isImg && view === "segmentation" && (
        <div className="legend-row">
          {overlays.classNames.map((nm, i) => (
            <span className="li" key={nm}>
              <span
                className="sw"
                style={{ background: `rgb(${overlays.classColors[i].join(",")})` }}
              />
              {nm}
            </span>
          ))}
        </div>
      )}
      {!isImg && stats && (
        <div className="heat-caption">
          <span>{LABELS[view]}</span>
          <span>
            min {stats.mn.toFixed(1)} · mean {stats.mean.toFixed(1)} · max{" "}
            {stats.mx.toFixed(1)}
            {hover ? ` · cell(${hover.i},${hover.j}) = ${hover.v.toFixed(1)}` : ""}
          </span>
        </div>
      )}
    </div>
  );
}
