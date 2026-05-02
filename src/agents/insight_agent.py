"""
Insight agent — synthesises risk + fraud findings into one coherent narrative.
"""

import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentContract, DeterminismConfig

SYSTEM = """You are a senior financial intelligence analyst.
You receive findings from a risk scoring specialist and a fraud pattern specialist.
Synthesise their findings into one clear, coherent narrative that a bank manager could read.
Highlight the most important signals. Do not repeat yourself. Keep it to 4-5 sentences."""

CONTRACT = AgentContract(
    agent_name="insight",
    version="1.0.0",
    schema_version="1.0.0",
    role="Financial intelligence synthesis analyst",
    objective="Combine specialist outputs into one coherent, decision-support narrative.",
    instructions="Prioritize the strongest evidence and synthesize without duplication.",
    skills=("cross-signal synthesis", "executive summarization", "evidence prioritization"),
    tools=("planner_output", "risk_output", "fraud_output", "llm_reasoning"),
    constraints=(
        "Do not introduce evidence that did not appear upstream.",
        "Keep output concise and readable for decision-makers.",
        "Output must be 4-5 sentences.",
    ),
    input_schema={
        "transaction_fields": "dict",
        "planner_output": "string",
        "risk_output": "string",
        "fraud_output": "string",
        "risk_score": "float",
        "anomaly_score": "float",
    },
    output_schema={"format": "plain text narrative", "constraints": "4-5 sentences"},
)


def _validate_insight_output(output: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]+", output) if s.strip()]
    errors: list[str] = []
    if not 4 <= len(sentences) <= 5:
        errors.append(f"Expected 4-5 sentences, got {len(sentences)}.")
    return errors


class InsightAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Insight",
            contract=CONTRACT,
            determinism=DeterminismConfig(temperature=0.15, top_p=0.9, max_retries=2, timeout_s=35.0),
        )

    def _prompt(self, tx, planner_output, risk_output, fraud_output, risk_score, anomaly_score) -> str:
        return f"""Transaction: ${tx.get('amount', 0):.2f} at {tx.get('merchant_type', 'unknown')} in {tx.get('location', 'unknown')} at {tx.get('hour', 12)}:00

Risk Score: {risk_score:.3f}
Anomaly Score: {anomaly_score:.4f}

Planner identified:
{planner_output}

Risk specialist found:
{risk_output}

Fraud specialist found:
{fraud_output}

Synthesise all findings into one clear narrative for a bank manager."""

    def synthesize(self, tx, planner_output, risk_output, fraud_output, risk_score, anomaly_score) -> str:
        return self.run(
            self._prompt(tx, planner_output, risk_output, fraud_output, risk_score, anomaly_score),
            SYSTEM,
            max_tokens=500,
            output_validator=_validate_insight_output,
        )

    def synthesize_stream(self, tx, planner_output, risk_output, fraud_output, risk_score, anomaly_score):
        return self.run_stream(
            self._prompt(tx, planner_output, risk_output, fraud_output, risk_score, anomaly_score),
            SYSTEM,
            max_tokens=500,
            output_validator=_validate_insight_output,
        )
