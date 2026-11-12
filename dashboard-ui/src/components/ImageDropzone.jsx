import React, { useRef, useState } from "react";

const STAGES = [
  "Layer 0 · Vision segmentation & buildable masks",
  "Layer 1 · Score-based data assimilation (NIWE masts)",
  "Layer 2 · GraphCast/Aurora → CorrDiff 2 km → ABL-PINN",
  "Layer 3 · PostGIS exclusions + Fuzzy Spatial GNN",
  "Layer 4 · Report synthesis & monitoring thread",
];

export default function ImageDropzone({ onAnalyzed, onError }) {
  const fileRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(-1);
  const [lat, setLat] = useState("14.2250");
  const [lon, setLon] = useState("76.3980");
  const [fileName, setFileName] = useState(null);

  async function analyze(file) {
    if (!file || busy) return;
    setBusy(true);
    setFileName(file.name);
    onError(null);
    let s = 0;
    setStage(0);
    const ticker = setInterval(() => {
      s = Math.min(s + 1, STAGES.length - 1);
      setStage(s);
    }, 900);
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (lat && lon) {
        fd.append("pinLat", lat);
        fd.append("pinLon", lon);
      }
      fd.append("gsdM", "10");
      const res = await fetch("/api/v1/analyze-land-image", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const payload = await res.json();
      const url = URL.createObjectURL(file);
      onAnalyzed({ ...payload, originalUrl: url, fileName: file.name });
    } catch (e) {
      onError(String(e.message || e));
    } finally {
      clearInterval(ticker);
      setStage(-1);
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>
        Land Image Ingestion <span className="tag">LAYER 0</span>
      </h2>
      <div
        className={`dropzone ${drag ? "drag" : ""} ${busy ? "busy" : ""}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          analyze(e.dataTransfer.files?.[0]);
        }}
      >
        <div className="dz-scan" />
        <div className="dz-icon">🛰️</div>
        <p>{busy ? "Analyzing parcel…" : "Drop a land image, or click to browse"}</p>
        <small>
          GeoTIFF · GPS-tagged drone JPEG · aerial PNG screenshot
          {fileName ? ` — ${fileName}` : ""}
        </small>
        <input
          ref={fileRef}
          type="file"
          accept=".tif,.tiff,.jpg,.jpeg,.png"
          hidden
          onChange={(e) => analyze(e.target.files?.[0])}
        />
      </div>

      <div className="pin-row">
        <input
          value={lat}
          onChange={(e) => setLat(e.target.value)}
          placeholder="Map-pin latitude"
          aria-label="fallback latitude"
        />
        <input
          value={lon}
          onChange={(e) => setLon(e.target.value)}
          placeholder="Map-pin longitude"
          aria-label="fallback longitude"
        />
      </div>

      {busy && (
        <div className="cascade">
          {STAGES.map((label, i) => (
            <div
              key={label}
              className={`stage ${i === stage ? "active" : i < stage ? "done" : ""}`}
            >
              <span className="dot" />
              {label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
