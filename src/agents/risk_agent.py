"""
Risk agent — interprets XGBoost risk score in plain language.
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentContract, DeterminismConfig
from config import LLM_FAST

SYSTEM = """You are a risk scoring specialist at a bank.
You receive a machine-learning risk score (0.0 = definitely safe, 1.0 = definitely fraud)
and interpret what it means in the context of the specific transaction.
Be specific about which transaction features drove the score. Keep it to 2-3 sentences."""

CONTRACT = AgentContract(
    agent_name="risk",
    version="1.0.0",
    schema_version="1.0.0",
    role="Risk scoring interpreter",
    objective="Translate model risk score into transparent, transaction-specific rationale.",
    instructions="Explain which features most influenced risk and keep response concise.",
    skills=("risk interpretation", "feature attribution narration", "financial context mapping"),
    tools=("risk_model_score", "llm_reasoning"),
    constraints=(
        "Do not issue final decision labels.",
        "Do not rely on data outside prompt inputs.",
        "Output must be 2-3 sentences.",
    ),
    input_schema={
        "transaction_fields": "dict",
        "risk_score": "float in [0,1]",
    },
    output_schema={"format": "plain text", "constraints": "2-3 sentences"},
)


def _validate_risk_output(output: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
    errors: list[str] = []
    if not 2 <= len(sentences) <= 3:
        errors.append(f"Expected 2-3 sentences, got {len(sentences)}.")
    return errors


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model=LLM_FAST,
            name="Risk",
            contract=CONTRACT,
            determinism=DeterminismConfig(temperature=0.1, top_p=0.9, max_retries=2, timeout_s=30.0),
        )

    def _prompt(self, tx: dict, risk_score: float) -> str:
        level = "LOW" if risk_score < 0.4 else "MEDIUM" if risk_score < 0.7 else "HIGH"
        return f"""Transaction details:
- Amount: ${tx.get('amount', 0):.2f} (user average ratio: {tx.get('amount_vs_avg', 1.0):.2f}x)
- Merchant type: {tx.get('merchant_type', 'unknown')}
- Location: {tx.get('location', 'unknown')}, home city: {bool(tx.get('is_home_city', 1))}
- Hour: {tx.get('hour', 12)}:00, late night: {bool(tx.get('is_late_night', 0))}
- Known merchant: {bool(tx.get('is_known_merchant', 1))}

ML Risk Score: {risk_score:.3f} ({level} RISK)

Interpret this risk score in the context of the transaction. What specific features contributed most?"""

    def interpret(self, tx: dict, risk_score: float) -> str:
        return self.run(self._prompt(tx, risk_score), SYSTEM, output_validator=_validate_risk_output)

    def interpret_stream(self, tx: dict, risk_score: float):
        return self.run_stream(self._prompt(tx, risk_score), SYSTEM, output_validator=_validate_risk_output)
