import React, { useState } from "react";
import AssistantPanel from "./components/AssistantPanel.jsx";
import ImageDropzone from "./components/ImageDropzone.jsx";
import MapViewport from "./components/MapViewport.jsx";
import ReportViewer from "./components/ReportViewer.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  return (
    <>
      <div className="aurora" aria-hidden />
      <div className="gridlines" aria-hidden />
      <div className="shell">
        <header className="topbar">
          <div className="logo">🌍</div>
          <div className="title-wrap">
            <h1>TerraGuard-v2 · Renewable Site Vision AI</h1>
            <p>image → micro-climate → suitability → investor report</p>
          </div>
          <div className="spacer" />
          <span className="pill">
            <span className="pulse-dot" /> MULTI-AGENT CASCADE · L0-L4
          </span>
        </header>

        <main className="grid">
          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            <ImageDropzone onAnalyzed={setResult} onError={setError} />
            {error && <div className="err">✖ {error}</div>}
            <MapViewport result={result} />
          </div>
          <ReportViewer result={result} />
        </main>
      </div>
      <AssistantPanel threadId={result?.report?.reportId} />
    </>
  );
}
