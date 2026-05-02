"""
Evaluator agent — final decision maker: APPROVE / FLAG / BLOCK.
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentContract, DeterminismConfig

SYSTEM = """You are the final decision authority for transaction fraud review at a bank.
Based on all the evidence provided, you must output EXACTLY this format and nothing else:

DECISION: <APPROVE|FLAG|BLOCK>
CONFIDENCE: <0.0-1.0>
REASON: <one sentence explaining the decision>

Rules:
- APPROVE: low risk, normal transaction, confidence ≥ 0.7
- FLAG: uncertain, needs human review, medium risk
- BLOCK: clear fraud signals, high risk, protect the customer
Do not add any other text."""

CONTRACT = AgentContract(
    agent_name="evaluator",
    version="1.0.0",
    schema_version="1.0.0",
    role="Final transaction decision authority",
    objective="Produce deterministic approve/flag/block decision with calibrated confidence.",
    instructions="Follow the output template exactly and justify decision in one sentence.",
    skills=("decision arbitration", "evidence weighting", "risk governance"),
    tools=("risk_score", "anomaly_score", "mcts_action", "insight_narrative"),
    constraints=(
        "Output must match DECISION/CONFIDENCE/REASON template exactly.",
        "DECISION must be one of APPROVE, FLAG, BLOCK.",
        "CONFIDENCE must be numeric in [0,1].",
    ),
    input_schema={
        "transaction_fields": "dict",
        "risk_score": "float",
        "anomaly_score": "float",
        "is_anomaly": "bool",
        "mcts_action": "string",
        "insight": "string",
    },
    output_schema={
        "DECISION": "APPROVE|FLAG|BLOCK",
        "CONFIDENCE": "float 0.0-1.0",
        "REASON": "one sentence",
    },
)


def _validate_evaluator_output(output: str) -> list[str]:
    errors: list[str] = []
    if not re.search(r"DECISION:\s*(APPROVE|FLAG|BLOCK)\b", output, re.IGNORECASE):
        errors.append("Missing or invalid DECISION field.")
    if not re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE):
        errors.append("Missing CONFIDENCE field.")
    if not re.search(r"REASON:\s*.+", output, re.IGNORECASE):
        errors.append("Missing REASON field.")
    return errors


class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Evaluator",
            contract=CONTRACT,
            determinism=DeterminismConfig(temperature=0.0, top_p=1.0, max_retries=2, timeout_s=30.0),
        )

    def _prompt(self, tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight) -> str:
        return f"""Transaction summary:
- Amount: ${tx.get('amount', 0):.2f}, merchant: {tx.get('merchant_type', 'unknown')}
- Location: {tx.get('location', 'unknown')}, home city: {bool(tx.get('is_home_city', 1))}
- Hour: {tx.get('hour', 12)}:00, late night: {bool(tx.get('is_late_night', 0))}

Scores:
- ML Risk Score: {risk_score:.3f} (0=safe, 1=fraud)
- Anomaly Score: {anomaly_score:.4f} (below -0.2 = anomalous)
- Is Statistical Anomaly: {is_anomaly}
- MCTS Recommended Action: {mcts_action}

Full analysis:
{insight}

Make the final decision."""

    def decide(self, tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight) -> dict:
        raw = self.run(
            self._prompt(tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight),
            SYSTEM,
            max_tokens=180,
            output_validator=_validate_evaluator_output,
        )
        return self._parse(raw, mcts_action)

    def decide_stream(self, tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight):
        return self.run_stream(
            self._prompt(tx, risk_score, anomaly_score, is_anomaly, mcts_action, insight),
            SYSTEM,
            max_tokens=180,
            output_validator=_validate_evaluator_output,
        )

    def _parse(self, raw: str, fallback_action: str) -> dict:
        parse_errors: list[str] = []
        decision = "FLAG"
        confidence = 0.5
        reason = raw

        m = re.search(r"DECISION:\s*(APPROVE|FLAG|BLOCK)", raw, re.IGNORECASE)
        if m:
            decision = m.group(1).upper()
        else:
            parse_errors.append("DECISION")

        m = re.search(r"CONFIDENCE:\s*([0-9.]+)", raw, re.IGNORECASE)
        if m:
            confidence = min(1.0, max(0.0, float(m.group(1))))
        else:
            parse_errors.append("CONFIDENCE")

        m = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
        else:
            parse_errors.append("REASON")

        if parse_errors:
            fallback = fallback_action.upper() if fallback_action else "FLAG"
            decision = fallback if fallback in {"APPROVE", "FLAG", "BLOCK"} else "FLAG"
            confidence = min(confidence, 0.35)
            reason = (
                "Evaluator output was non-compliant; downgraded to uncertain decision path and "
                "requires human review."
            )

        return {"decision": decision, "confidence": confidence, "reason": reason}
