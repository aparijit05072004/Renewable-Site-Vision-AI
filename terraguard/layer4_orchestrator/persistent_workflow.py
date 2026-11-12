"""Cyclic multi-agent workflow with durable checkpointing (blueprint §8).

Production targets LangGraph ``StateGraph`` + ``PostgresSaver`` on
PostgreSQL 16 (docker-compose.yml provisions it).  To keep the pipeline
runnable with zero external services, this module implements the same
execution semantics — cyclic node graph, append-only message reduction,
``interrupt()``-style HITL pauses, thread-scoped durable checkpoints,
lossless resume after process restarts — over a SQLite-backed saver with
the PostgresSaver interface.  If ``langgraph`` + a Postgres DSN are
available at runtime, ``build_langgraph_workflow`` wires the identical
nodes into a real LangGraph instead.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from ..config import DATA_DIR
from . import agents
from .state import ParcelMonitoringState, initial_state


class SqliteSaver:
    """PostgresSaver-compatible durable checkpointer (thread-scoped)."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DATA_DIR / "terraguard_checkpoints.db")
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT, ts REAL, state_json TEXT,
                PRIMARY KEY (thread_id, ts))""")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def put(self, thread_id: str, state: ParcelMonitoringState) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?)",
                      (thread_id, time.time(), json.dumps(dict(state))))

    def get(self, thread_id: str) -> ParcelMonitoringState | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT state_json FROM checkpoints WHERE thread_id=? "
                "ORDER BY ts DESC LIMIT 1", (thread_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, thread_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, state_json FROM checkpoints WHERE thread_id=? "
                "ORDER BY ts", (thread_id,)).fetchall()
        return [{"ts": r[0], "state": json.loads(r[1])} for r in rows]

    def list_threads(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        return [r[0] for r in rows]


class PersistentParcelWorkflow:
    """Cyclic graph: meteo -> geo -> financial -> hitl -> (loop | wait | end)."""

    NODES = ("meteorological_drift", "geospatial_regulatory",
             "financial_orchestrator", "hitl_gate")

    def __init__(self, saver: SqliteSaver | None = None):
        self.saver = saver or SqliteSaver()

    # ------------------------------------------------------------------
    def register_parcel(self, parcel_summary: dict) -> str:
        thread_id = f"parcel-{uuid.uuid4().hex[:12]}"
        state = initial_state(
            parcel_id=thread_id,
            buildable_ha=parcel_summary["netBuildableHa"],
            buildable_pct=parcel_summary["netBuildablePct"],
            cloud=parcel_summary["meanCloudCover"],
            wpd=parcel_summary["meanWpd120m"],
            irr_pct=parcel_summary["irrPct"],
            suitability=parcel_summary["suitabilityScore"])
        state["messages"].append({
            "role": "system",
            "content": f"[Orchestrator] Monitoring thread {thread_id} "
                       f"registered with durable checkpointing."})
        self.saver.put(thread_id, state)
        return thread_id

    # ------------------------------------------------------------------
    def _apply(self, state: ParcelMonitoringState, update: dict) -> None:
        msgs = update.pop("messages", [])
        state["messages"] = list(state.get("messages", [])) + msgs
        state.update(update)

    def step(self, thread_id: str,
             sensor_update: dict | None = None) -> ParcelMonitoringState:
        """Run one monitoring cycle; checkpoints after every node."""
        state = self.saver.get(thread_id)
        if state is None:
            raise KeyError(f"unknown thread {thread_id}")
        if sensor_update:
            state.update(sensor_update)
        if state.get("hitlPending") and state.get("stakeholderDecision") is None:
            return state  # interrupted — graph frozen until HITL resume
        state["cycle"] = int(state.get("cycle", 0)) + 1
        for name, fn in (("meteorological_drift", agents.meteorological_drift_node),
                         ("geospatial_regulatory", agents.geospatial_regulatory_node),
                         ("financial_orchestrator", agents.financial_orchestrator_node),
                         ("hitl_gate", agents.hitl_gate_node)):
            self._apply(state, fn(state))
            self.saver.put(thread_id, state)   # durable after each node
        return state

    # ------------------------------------------------------------------
    def resume_hitl(self, thread_id: str, decision: str,
                    override_thresholds: dict | None = None
                    ) -> ParcelMonitoringState:
        """HITL Command(resume=...) — stakeholder sign-off console."""
        state = self.saver.get(thread_id)
        if state is None:
            raise KeyError(f"unknown thread {thread_id}")
        state["stakeholderDecision"] = decision
        state["hitlPending"] = False
        if override_thresholds:
            state.update(override_thresholds)
        state["messages"] = list(state.get("messages", [])) + [{
            "role": "user",
            "content": f"[HITL Console] Stakeholder decision: {decision}"}]
        self.saver.put(thread_id, state)
        return state


def build_langgraph_workflow(postgres_dsn: str):  # pragma: no cover
    """Wire the identical nodes into LangGraph + PostgresSaver when available."""
    from langgraph.checkpoint.postgres import PostgresSaver      # type: ignore
    from langgraph.graph import END, START, StateGraph           # type: ignore

    graph = StateGraph(ParcelMonitoringState)
    graph.add_node("meteorological_drift", agents.meteorological_drift_node)
    graph.add_node("geospatial_regulatory", agents.geospatial_regulatory_node)
    graph.add_node("financial_orchestrator", agents.financial_orchestrator_node)
    graph.add_node("hitl_gate", agents.hitl_gate_node)
    graph.add_edge(START, "meteorological_drift")
    graph.add_edge("meteorological_drift", "geospatial_regulatory")
    graph.add_edge("geospatial_regulatory", "financial_orchestrator")
    graph.add_edge("financial_orchestrator", "hitl_gate")
    graph.add_conditional_edges("hitl_gate", agents.route_after_hitl,
                                {"WAIT_HUMAN": END, "END": END,
                                 "CONTINUE": "meteorological_drift"})
    saver = PostgresSaver.from_conn_string(postgres_dsn)
    return graph.compile(checkpointer=saver)
