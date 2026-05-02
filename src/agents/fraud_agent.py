"""
Fraud agent — matches transaction against retrieved fraud patterns.
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentContract, DeterminismConfig
from config import LLM_FAST

SYSTEM = """You are a fraud pattern specialist at a bank.
You receive a transaction, its anomaly score from an Isolation Forest model,
and a list of relevant fraud rules retrieved from the knowledge base.
Your job: identify which (if any) fraud patterns this transaction matches and explain why.
Be specific. Keep it to 3-4 sentences."""

CONTRACT = AgentContract(
    agent_name="fraud",
    version="1.0.0",
    schema_version="1.0.0",
    role="Fraud pattern specialist",
    objective="Map transaction signals to known fraud typologies with explicit evidence.",
    instructions="Match evidence to retrieved rules and describe pattern alignment concisely.",
    skills=("pattern matching", "anomaly interpretation", "rule-grounded explanation"),
    tools=("anomaly_score", "rag_rule_retrieval", "llm_reasoning"),
    constraints=(
        "Do not produce final approve/flag/block decisions.",
        "Do not reference fraud rules that were not retrieved.",
        "Output must be 3-4 sentences.",
    ),
    input_schema={
        "transaction_fields": "dict",
        "anomaly_score": "float",
        "is_anomaly": "bool",
        "rag_results": "list[rule]",
    },
    output_schema={"format": "plain text", "constraints": "3-4 sentences"},
)


def _validate_fraud_output(output: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
    errors: list[str] = []
    if not 3 <= len(sentences) <= 4:
        errors.append(f"Expected 3-4 sentences, got {len(sentences)}.")
    return errors


class FraudAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model=LLM_FAST,
            name="Fraud",
            contract=CONTRACT,
            determinism=DeterminismConfig(temperature=0.1, top_p=0.9, max_retries=2, timeout_s=30.0),
        )

    def _prompt(self, tx: dict, anomaly_score: float, is_anomaly: bool, rag_results: list[dict]) -> str:
        rules_text = "\n".join(
            f"- [{r['category']}] {r['text'][:120]}..." for r in rag_results
        )
        return f"""Transaction:
- Amount: ${tx.get('amount', 0):.2f}, merchant: {tx.get('merchant_type', 'unknown')}
- Location: {tx.get('location', 'unknown')}, home city: {bool(tx.get('is_home_city', 1))}
- Hour: {tx.get('hour', 12)}:00, late night: {bool(tx.get('is_late_night', 0))}
- Known merchant: {bool(tx.get('is_known_merchant', 1))}
- Amount vs avg: {tx.get('amount_vs_avg', 1.0):.2f}x

Anomaly Score: {anomaly_score:.4f} (is_anomaly: {is_anomaly})
(Score < -0.2 means statistically abnormal behaviour)

Most relevant fraud rules from knowledge base:
{rules_text}

Which fraud patterns does this transaction match? Explain the match specifically."""

    def analyze(self, tx: dict, anomaly_score: float, is_anomaly: bool, rag_results: list[dict]) -> str:
        return self.run(
            self._prompt(tx, anomaly_score, is_anomaly, rag_results),
            SYSTEM,
            output_validator=_validate_fraud_output,
        )

    def analyze_stream(self, tx: dict, anomaly_score: float, is_anomaly: bool, rag_results: list[dict]):
        return self.run_stream(
            self._prompt(tx, anomaly_score, is_anomaly, rag_results),
            SYSTEM,
            output_validator=_validate_fraud_output,
        )
