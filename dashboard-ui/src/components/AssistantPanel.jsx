import React, { useEffect, useRef, useState } from "react";

const WELCOME =
  "Hey, I'm TERRA ⚡ — the TerraGuard site-intelligence copilot.\n\n" +
  "Upload a land parcel and I can break down its wind & solar physics, the " +
  "GNN suitability verdict, 25-year financials, drift risks — or explain any " +
  "concept in the stack (CorrDiff, PINN, SDA, Weibull, LCOE…).";

const CHIPS = [
  "Summarize this site",
  "Wind or solar here?",
  "Explain the financials",
  "Why this score?",
  "What is CorrDiff?",
  "Monitoring status",
];

export default function AssistantPanel({ threadId }) {
  const [openPanel, setOpenPanel] = useState(false);
  const [msgs, setMsgs] = useState([{ role: "bot", text: WELCOME }]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const bodyRef = useRef(null);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [msgs, typing, openPanel]);

  async function send(text) {
    const q = (text ?? input).trim();
    if (!q || typing) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", text: q }]);
    setTyping(true);
    try {
      const res = await fetch("/api/v1/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q, threadId }),
      });
      const j = await res.json();
      await typeOut(j.reply || "…");
    } catch {
      await typeOut("⚠ I lost contact with the TerraGuard engine — is the API server running on :8000?");
    } finally {
      setTyping(false);
    }
  }

  function typeOut(full) {
    return new Promise((resolve) => {
      setMsgs((m) => [...m, { role: "bot", text: "" }]);
      let i = 0;
      const step = Math.max(2, Math.round(full.length / 90));
      const iv = setInterval(() => {
        i = Math.min(full.length, i + step);
        setMsgs((m) => {
          const copy = m.slice();
          copy[copy.length - 1] = { role: "bot", text: full.slice(0, i) };
          return copy;
        });
        if (i >= full.length) {
          clearInterval(iv);
          resolve();
        }
      }, 16);
    });
  }

  return (
    <>
      {!openPanel && <div className="orb-ring" aria-hidden />}
      <button
        className="orb"
        onClick={() => setOpenPanel((o) => !o)}
        aria-label={openPanel ? "Close TERRA assistant" : "Open TERRA assistant"}
        title="TERRA — site intelligence assistant"
      >
        {openPanel ? "✕" : "⚡"}
      </button>

      {openPanel && (
        <div className="assistant" role="dialog" aria-label="TERRA assistant">
          <div className="as-head">
            <div className="as-avatar">⚡</div>
            <div>
              <div className="nm">TERRA</div>
              <div className="st">
                {typing ? (
                  <>
                    thinking
                    <span className="wave"><i /><i /><i /><i /><i /></span>
                  </>
                ) : (
                  <>online · grounded in {threadId ? "live report" : "no report yet"}</>
                )}
              </div>
            </div>
            <button className="close" onClick={() => setOpenPanel(false)} aria-label="close">
              ✕
            </button>
          </div>

          <div className="as-body" ref={bodyRef}>
            {msgs.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                {m.text}
              </div>
            ))}
            {typing && (
              <div className="msg bot typing">
                <i /><i /><i />
              </div>
            )}
          </div>

          <div className="chips">
            {CHIPS.map((c) => (
              <button key={c} onClick={() => send(c)}>
                {c}
              </button>
            ))}
          </div>

          <div className="as-input">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Ask TERRA about this site…"
              aria-label="message TERRA"
            />
            <button onClick={() => send()} aria-label="send">➤</button>
          </div>
        </div>
      )}
    </>
  );
}
