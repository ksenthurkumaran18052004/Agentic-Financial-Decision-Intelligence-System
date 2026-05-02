"""
Orchestrator — runs the full multi-agent pipeline for one transaction.

Flow:
  transaction
    → ML scores (risk + anomaly)
    → RAG retrieval
    → MCTS recommends action
    → Planner + Risk + Fraud agents (parallel intent, sequential here for simplicity)
    → Insight agent synthesises
    → Evaluator makes final call
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import risk_model, fraud_model
from src.rag.retriever import retrieve_for_transaction
from src.reasoning.mcts import MCTSReasoner
from src.agents.planner_agent import PlannerAgent
from src.agents.risk_agent import RiskAgent
from src.agents.fraud_agent import FraudAgent
from src.agents.insight_agent import InsightAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.base_agent import AgentRunTrace

_planner   = None
_risk_ag   = None
_fraud_ag  = None
_insight   = None
_evaluator = None


def _agents():
    global _planner, _risk_ag, _fraud_ag, _insight, _evaluator
    if _planner is None:
        _planner   = PlannerAgent()
        _risk_ag   = RiskAgent()
        _fraud_ag  = FraudAgent()
        _insight   = InsightAgent()
        _evaluator = EvaluatorAgent()


def analyze(tx: dict) -> dict:
    """
    Run the full pipeline on a transaction dict.
    Returns a structured result with decision, scores, and agent reasoning.
    """
    _agents()
    t0 = time.time()

    # --- ML scoring ---
    risk_score    = risk_model.predict(tx)
    anomaly_score, is_anomaly = fraud_model.predict(tx)

    # --- RAG retrieval ---
    rag_results = retrieve_for_transaction(tx, top_k=4)

    # --- MCTS reasoning ---
    mcts = MCTSReasoner(risk_score=risk_score, anomaly_score=anomaly_score, rag_hits=len(rag_results))
    mcts_action, mcts_simulations = mcts.search()

    # --- Agent chain ---
    # Validate inputs before invoking agents. If validation fails, produce
    # an UNCERTAIN_OUTPUT for that agent and attach validation errors to the
    # agent's run trace so the orchestrator and callers can react accordingly.

    # Planner
    planner_errors = _planner.validate_inputs(tx)
    if planner_errors:
        planner_out = (
            f"UNCERTAIN_OUTPUT: planner input validation failed: {'; '.join(planner_errors)}"
        )
        _planner.last_trace = AgentRunTrace(
            agent_name=_planner.contract.agent_name,
            contract_version=_planner.contract.version,
            schema_version=_planner.contract.schema_version,
            model=_planner.model,
            latency_ms=0,
            retries=0,
            output_valid=False,
            validation_errors=planner_errors,
        )
    else:
        planner_out = _planner.plan(tx)

    # Risk
    risk_inputs = {"transaction_fields": tx, "risk_score": risk_score}
    risk_errors = _risk_ag.validate_inputs(risk_inputs)
    if risk_errors:
        risk_out = f"UNCERTAIN_OUTPUT: risk input validation failed: {'; '.join(risk_errors)}"
        _risk_ag.last_trace = AgentRunTrace(
            agent_name=_risk_ag.contract.agent_name,
            contract_version=_risk_ag.contract.version,
            schema_version=_risk_ag.contract.schema_version,
            model=_risk_ag.model,
            latency_ms=0,
            retries=0,
            output_valid=False,
            validation_errors=risk_errors,
        )
    else:
        risk_out = _risk_ag.interpret(tx, risk_score)

    # Fraud
    fraud_inputs = {
        "transaction_fields": tx,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "rag_results": rag_results,
    }
    fraud_errors = _fraud_ag.validate_inputs(fraud_inputs)
    if fraud_errors:
        fraud_out = f"UNCERTAIN_OUTPUT: fraud input validation failed: {'; '.join(fraud_errors)}"
        _fraud_ag.last_trace = AgentRunTrace(
            agent_name=_fraud_ag.contract.agent_name,
            contract_version=_fraud_ag.contract.version,
            schema_version=_fraud_ag.contract.schema_version,
            model=_fraud_ag.model,
            latency_ms=0,
            retries=0,
            output_valid=False,
            validation_errors=fraud_errors,
        )
    else:
        fraud_out = _fraud_ag.analyze(tx, anomaly_score, is_anomaly, rag_results)

    # Insight
    insight_inputs = {
        "transaction_fields": tx,
        "planner_output": planner_out,
        "risk_output": risk_out,
        "fraud_output": fraud_out,
        "risk_score": risk_score,
        "anomaly_score": anomaly_score,
    }
    insight_errors = _insight.validate_inputs(insight_inputs)
    if insight_errors:
        insight_out = f"UNCERTAIN_OUTPUT: insight input validation failed: {'; '.join(insight_errors)}"
        _insight.last_trace = AgentRunTrace(
            agent_name=_insight.contract.agent_name,
            contract_version=_insight.contract.version,
            schema_version=_insight.contract.schema_version,
            model=_insight.model,
            latency_ms=0,
            retries=0,
            output_valid=False,
            validation_errors=insight_errors,
        )
    else:
        insight_out = _insight.synthesize(tx, planner_out, risk_out, fraud_out, risk_score, anomaly_score)

    # Evaluator
    evaluator_inputs = {
        "transaction_fields": tx,
        "risk_score": risk_score,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "mcts_action": mcts_action,
        "insight": insight_out,
    }
    evaluator_errors = _evaluator.validate_inputs(evaluator_inputs)
    if evaluator_errors:
        eval_out = {
            "decision": "FLAG",
            "confidence": 0.0,
            "reason": f"UNCERTAIN_OUTPUT: evaluator input validation failed: {'; '.join(evaluator_errors)}",
        }
        _evaluator.last_trace = AgentRunTrace(
            agent_name=_evaluator.contract.agent_name,
            contract_version=_evaluator.contract.version,
            schema_version=_evaluator.contract.schema_version,
            model=_evaluator.model,
            latency_ms=0,
            retries=0,
            output_valid=False,
            validation_errors=evaluator_errors,
        )
    else:
        eval_out = _evaluator.decide(tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight_out)

    elapsed = round(time.time() - t0, 2)

    agent_contracts = {
        "planner": _planner.contract_metadata(),
        "risk": _risk_ag.contract_metadata(),
        "fraud": _fraud_ag.contract_metadata(),
        "insight": _insight.contract_metadata(),
        "evaluator": _evaluator.contract_metadata(),
    }
    agent_evaluations = {
        "planner": _planner.latest_evaluation(),
        "risk": _risk_ag.latest_evaluation(),
        "fraud": _fraud_ag.latest_evaluation(),
        "insight": _insight.latest_evaluation(),
        "evaluator": _evaluator.latest_evaluation(),
    }

    return {
        "transaction_id": tx.get("transaction_id", "unknown"),
        "decision":       eval_out["decision"],
        "confidence":     eval_out["confidence"],
        "reason":         eval_out["reason"],
        "risk_score":     round(risk_score, 4),
        "anomaly_score":  round(anomaly_score, 4),
        "is_anomaly":     is_anomaly,
        "mcts_action":    mcts_action,
        "mcts_simulations": mcts_simulations,
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
        "agent_contracts": agent_contracts,
        "agent_evaluations": agent_evaluations,
        "processing_time_s": elapsed,
    }
