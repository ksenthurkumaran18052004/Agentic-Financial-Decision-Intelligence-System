"""
Live streaming fraud detection dashboard with agent "thinking" display.
Run: streamlit run src/dashboard/app.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="AFDIS — Live Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
)

DECISION_COLOR = {"APPROVE": "#22c55e", "FLAG": "#f59e0b", "BLOCK": "#ef4444"}
DECISION_BG    = {"APPROVE": "#dcfce7", "FLAG": "#fef9c3", "BLOCK": "#fee2e2"}
DECISION_ICON  = {"APPROVE": "✅", "FLAG": "⚠️", "BLOCK": "🚫"}


# ── Session state ────────────────────────────────────────────────────────────

def _init():
    for k, v in {"streaming": False, "results": [], "fraud_rate": 0.20}.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Charts ───────────────────────────────────────────────────────────────────

def _donut(results):
    counts = {k: sum(1 for r in results if r["decision"] == k) for k in DECISION_COLOR}
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.6,
        marker_colors=list(DECISION_COLOR.values()),
        textinfo="label+percent",
        showlegend=False,
    ))
    fig.update_layout(
        title={"text": "Decision Split", "x": 0.5, "font": {"size": 13}},
        margin=dict(t=40, b=5, l=5, r=5), height=210,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _timeline(results):
    if not results:
        return go.Figure()
    df = pd.DataFrame([
        {"i": i + 1, "risk": r["risk_score"], "dec": r["decision"], "id": r["transaction_id"]}
        for i, r in enumerate(results[-30:])
    ])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["i"], y=df["risk"], mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(color=[DECISION_COLOR[d] for d in df["dec"]], size=9),
        hovertemplate="%{customdata}<br>Risk: %{y:.3f}<extra></extra>",
        customdata=df["id"],
    ))
    fig.add_hline(y=0.6, line_dash="dash", line_color="#ef4444",
                  annotation_text="Risk threshold")
    fig.update_layout(
        title={"text": "Risk Score Timeline", "x": 0, "font": {"size": 13}},
        yaxis=dict(range=[0, 1]),
        height=210, margin=dict(t=40, b=30, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _format_broker_status(status: dict) -> str:
    if not status:
        return "Unavailable"
    mode = status.get("mode", "unknown")
    connected = status.get("connected", False)
    label = "Connected" if connected else "Disconnected"
    if mode == "memory":
        return "Local fallback"
    return f"{label} · Redis Streams"


# ── Core: analyze one transaction with full thinking display ─────────────────

def analyze_with_thinking(tx: dict) -> dict:
    """
    Runs the full agent pipeline inside an st.status() block.
    Each agent streams its tokens live — looks like Claude thinking.
    """
    from src.models import risk_model, fraud_model
    from src.rag.retriever import retrieve_for_transaction
    from src.reasoning.mcts import MCTSReasoner
    from src.agents.planner_agent import PlannerAgent
    from src.agents.risk_agent import RiskAgent
    from src.agents.fraud_agent import FraudAgent
    from src.agents.insight_agent import InsightAgent
    from src.agents.evaluator_agent import EvaluatorAgent
    from src.data.transaction_broker import get_transaction_broker

    planner_ag   = PlannerAgent()
    risk_ag      = RiskAgent()
    fraud_ag     = FraudAgent()
    insight_ag   = InsightAgent()
    evaluator_ag = EvaluatorAgent()
    broker = get_transaction_broker()

    result = {}

    with st.status(
        f"🔍 Analyzing `{tx['transaction_id']}` — "
        f"${tx['amount']:.2f} @ **{tx['merchant_type']}** in **{tx['location']}** at {tx['hour']}:00",
        expanded=True,
    ) as status:

        # ── Step 1: ML models ────────────────────────────────────────────────
        st.markdown("#### ⚙️ Step 1 — ML Scoring")
        risk_score             = risk_model.predict(tx)
        anomaly_score, is_anomaly = fraud_model.predict(tx)
        risk_lvl = "🔴 HIGH" if risk_score > 0.7 else "🟡 MEDIUM" if risk_score > 0.4 else "🟢 LOW"
        anom_lbl = "🔴 ANOMALOUS" if is_anomaly else "🟢 Normal"
        st.markdown(
            f"<span style='color:#6b7280; font-size:0.9em'>"
            f"→ XGBoost risk score: **{risk_score:.3f}** {risk_lvl}<br>"
            f"→ Isolation Forest:  **{anomaly_score:.4f}** {anom_lbl}"
            f"</span>",
            unsafe_allow_html=True,
        )

        # ── Step 2: RAG retrieval ────────────────────────────────────────────
        st.markdown("#### 📚 Step 2 — Knowledge Base Search")
        rag_results = retrieve_for_transaction(tx, top_k=4)
        for r in rag_results:
            st.markdown(
                f"<span style='color:#6b7280; font-size:0.85em'>"
                f"→ Matched rule `{r['category']}` &nbsp; relevance **{r['relevance_score']:.3f}**"
                f"</span>",
                unsafe_allow_html=True,
            )

        # ── Step 3: MCTS ─────────────────────────────────────────────────────
        st.markdown("#### 🌳 Step 3 — MCTS Decision Simulation")
        mcts = MCTSReasoner(risk_score=risk_score, anomaly_score=anomaly_score, rag_hits=len(rag_results))
        mcts_action, n_sims = mcts.search()
        mcts_summary = mcts.summary()
        mcts_row = "  ".join(
            f"`{a}` {d['visits']} visits · reward {d['mean_reward']:.3f}"
            for a, d in mcts_summary.items()
        )
        st.markdown(
            f"<span style='color:#6b7280; font-size:0.85em'>"
            f"→ {n_sims} simulations &nbsp;|&nbsp; {mcts_row}<br>"
            f"→ Recommended: **{mcts_action}**"
            f"</span>",
            unsafe_allow_html=True,
        )

        stream_status = broker.health()
        st.caption(f"Stream broker: {_format_broker_status(stream_status)}")

        # ── Step 4: Planner agent ────────────────────────────────────────────
        st.markdown("#### 🤖 Agent 1 — Planner")
        st.caption("Identifying what needs investigation...")
        planner_out = st.write_stream(planner_ag.plan_stream(tx))

        # ── Step 5: Risk agent ───────────────────────────────────────────────
        st.markdown("#### 🤖 Agent 2 — Risk Analyst")
        st.caption(f"Interpreting ML risk score {risk_score:.3f}...")
        risk_out = st.write_stream(risk_ag.interpret_stream(tx, risk_score))

        # ── Step 6: Fraud agent ──────────────────────────────────────────────
        st.markdown("#### 🤖 Agent 3 — Fraud Pattern Specialist")
        st.caption("Matching against retrieved fraud rules...")
        fraud_out = st.write_stream(
            fraud_ag.analyze_stream(tx, anomaly_score, is_anomaly, rag_results)
        )

        # ── Step 7: Insight agent ────────────────────────────────────────────
        st.markdown("#### 🧠 Agent 4 — Senior Insight Analyst")
        st.caption("Synthesising all findings...")
        insight_out = st.write_stream(
            insight_ag.synthesize_stream(tx, planner_out, risk_out, fraud_out, risk_score, anomaly_score)
        )

        # ── Step 8: Evaluator agent (stream + capture for parsing) ───────────
        st.markdown("#### ⚖️ Agent 5 — Evaluator (Final Decision)")
        st.caption("Making the final call...")

        tokens = []
        def _capture(gen):
            for t in gen:
                tokens.append(t)
                yield t

        st.write_stream(_capture(
            evaluator_ag.decide_stream(tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight_out)
        ))
        eval_out = evaluator_ag._parse("".join(tokens), mcts_action)

        # ── Final banner ─────────────────────────────────────────────────────
        dec    = eval_out["decision"]
        color  = DECISION_COLOR[dec]
        icon   = DECISION_ICON[dec]
        status.update(
            label=f"{icon} {dec} — {eval_out['confidence']:.0%} confidence — `{tx['transaction_id']}`",
            state="complete",
            expanded=False,
        )

    result = {
        "transaction_id": tx.get("transaction_id", "unknown"),
        "decision":       dec,
        "confidence":     eval_out["confidence"],
        "reason":         eval_out["reason"],
        "risk_score":     round(risk_score, 4),
        "anomaly_score":  round(anomaly_score, 4),
        "is_anomaly":     is_anomaly,
        "mcts_action":    mcts_action,
        "mcts_simulations": n_sims,
        "rag_rules_matched": [
            {"id": r["id"], "category": r["category"], "relevance": r["relevance_score"]}
            for r in rag_results
        ],
        "agent_reasoning": {
            "planner":   planner_out,
            "risk":      risk_out,
            "fraud":     fraud_out,
            "insight":   insight_out,
            "evaluator": eval_out["reason"],
        },
        "agent_contracts": {
            "planner": planner_ag.contract_metadata(),
            "risk": risk_ag.contract_metadata(),
            "fraud": fraud_ag.contract_metadata(),
            "insight": insight_ag.contract_metadata(),
            "evaluator": evaluator_ag.contract_metadata(),
        },
        "agent_evaluations": {
            "planner": planner_ag.latest_evaluation(),
            "risk": risk_ag.latest_evaluation(),
            "fraud": fraud_ag.latest_evaluation(),
            "insight": insight_ag.latest_evaluation(),
            "evaluator": evaluator_ag.latest_evaluation(),
        },
        # transaction fields for display
        "timestamp":     tx.get("timestamp", ""),
        "user_id":       tx.get("user_id", ""),
        "amount":        tx.get("amount", 0),
        "merchant_type": tx.get("merchant_type", ""),
        "location":      tx.get("location", ""),
        "hour":          tx.get("hour", 0),
        "stream_id":     tx.get("stream_id", ""),
        "stream_source": tx.get("stream_source", ""),
    }
    return result


# ── Page header ──────────────────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.title("🛡️ AFDIS — Live Fraud Detection")
    st.caption("Multi-agent AI · XGBoost · Isolation Forest · MCTS · LLM Reasoning (Together AI)")
with col_status:
    color = "#22c55e" if st.session_state.streaming else "#6b7280"
    label = "● LIVE" if st.session_state.streaming else "○ STOPPED"
    st.markdown(
        f"<div style='text-align:right;padding-top:20px'>"
        f"<span style='color:{color};font-weight:700;font-size:18px'>{label}</span></div>",
        unsafe_allow_html=True,
    )

# ── Controls ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns([1, 1, 2, 2])
with c1:
    if st.session_state.streaming:
        if st.button("⏹ Stop", use_container_width=True, type="secondary"):
            st.session_state.streaming = False
            st.rerun()
    else:
        if st.button("▶ Start Stream", use_container_width=True, type="primary"):
            st.session_state.streaming = True
            st.rerun()
with c2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.results = []
        st.rerun()
with c3:
    pct = st.slider("Fraud injection %", 5, 60, 20, 5, label_visibility="collapsed")
    st.caption(f"Fraud injection rate: {pct}%")
    st.session_state.fraud_rate = pct / 100

st.divider()

# ── Metrics ──────────────────────────────────────────────────────────────────

results = st.session_state.results
total   = len(results)
blocked = sum(1 for r in results if r["decision"] == "BLOCK")
flagged = sum(1 for r in results if r["decision"] == "FLAG")
approved= sum(1 for r in results if r["decision"] == "APPROVE")
frate   = (blocked + flagged) / max(total, 1)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Processed",   total)
m2.metric("Fraud Rate",  f"{frate:.1%}")
m3.metric("🚫 Blocked",  blocked)
m4.metric("⚠️ Flagged",  flagged)
m5.metric("✅ Approved", approved)

st.divider()

# ── Feed + Charts ─────────────────────────────────────────────────────────────

left, right = st.columns([3, 2])

with left:
    st.subheader("Transaction Feed")
    if not results:
        st.info("Press **▶ Start Stream** — transactions will appear here automatically.")
    else:
        rows = []
        for r in reversed(results[-20:]):
            rows.append({
                "Time":      r.get("timestamp", "")[-8:],
                "Txn":       r["transaction_id"],
                "User":      r.get("user_id", ""),
                "Amount":    f"${r['amount']:.2f}",
                "Merchant":  r.get("merchant_type", ""),
                "Location":  r.get("location", ""),
                "Risk":      f"{r['risk_score']:.0%}",
                "Decision":  f"{DECISION_ICON[r['decision']]} {r['decision']}",
                "Conf":      f"{r['confidence']:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)

with right:
    if results:
        st.plotly_chart(_donut(results),    use_container_width=True)
        st.plotly_chart(_timeline(results), use_container_width=True)
    else:
        st.caption("Charts appear after first transaction.")

st.divider()

# ── Latest transaction detail ─────────────────────────────────────────────────

if results:
    latest = results[-1]
    dec    = latest["decision"]
    color  = DECISION_COLOR[dec]
    icon   = DECISION_ICON[dec]
    st.markdown(
        f"<div style='background:{DECISION_BG[dec]};border-left:5px solid {color};"
        f"padding:12px 20px;border-radius:6px;margin-bottom:12px'>"
        f"<b>{icon} Last Decision: {dec}</b> &nbsp;·&nbsp; "
        f"<code>{latest['transaction_id']}</code> &nbsp;·&nbsp; "
        f"${latest['amount']:.2f} @ {latest['merchant_type']} in {latest['location']} "
        f"at {latest['hour']}:00 &nbsp;·&nbsp; Confidence: {latest['confidence']:.0%}<br>"
        f"<small style='color:#555'>{latest['reason']}</small></div>",
        unsafe_allow_html=True,
    )

# ── Streaming loop ────────────────────────────────────────────────────────────

if st.session_state.streaming:
    from src.data.live_stream import next_transaction
    from src.data.transaction_broker import get_transaction_broker

    broker = get_transaction_broker()
    
    # Debug: Show broker status
    health = broker.health()
    if health.get("mode") == "memory-fallback":
        st.warning(f"⚠️ Broker in fallback mode: {health.get('error', 'unknown')}")

    record = broker.take_next(seed_fn=lambda: next_transaction(fraud_rate=st.session_state.fraud_rate))
    tx = dict(record.transaction)
    tx["stream_id"] = record.stream_id
    tx["stream_source"] = record.source

    try:
        result = analyze_with_thinking(tx)
        result["stream_id"] = record.stream_id
        result["stream_source"] = record.source
        st.session_state.results.append(result)
        broker.ack(record.stream_id)
    except Exception:
        broker.ack(record.stream_id)
        raise

    time.sleep(1)
    st.rerun()
