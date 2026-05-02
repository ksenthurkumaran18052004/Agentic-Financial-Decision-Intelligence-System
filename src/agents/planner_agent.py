"""
Planner agent — reads raw transaction, identifies what to investigate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentContract, DeterminismConfig

SYSTEM = """You are a senior fraud analyst coordinator at a bank.
Your job is to read an incoming transaction and identify the key risk factors that need investigation.
Be concise. Output 3-5 bullet points of what specifically needs to be checked.
Do NOT make a final decision — only flag what needs analysis."""

CONTRACT = AgentContract(
    agent_name="planner",
    version="1.0.0",
    schema_version="1.0.0",
    role="Fraud analysis planner",
    objective="Identify the highest-priority risk factors for downstream analysis.",
    instructions="Read the transaction and produce concise investigation points.",
    skills=("transaction triage", "risk-factor decomposition", "context prioritization"),
    tools=("llm_reasoning",),
    constraints=(
        "Do not make final approve/flag/block decisions.",
        "Do not invent transaction fields not provided in input.",
        "Output must be 3-5 bullet points.",
    ),
    input_schema={
        "transaction_id": "string",
        "amount": "float",
        "merchant_type": "string",
        "location": "string",
        "hour": "integer 0-23",
        "is_home_city": "0|1",
        "is_known_merchant": "0|1",
        "is_late_night": "0|1",
        "amount_vs_avg": "float",
    },
    output_schema={"format": "plain text bullet list", "constraints": "3-5 bullet points"},
)


def _validate_planner_output(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    bullets = [line for line in lines if line.startswith("-") or line.startswith("*")]
    errors: list[str] = []
    if len(bullets) != len(lines):
        errors.append("Output must contain only bullet points.")
    if not 3 <= len(bullets) <= 5:
        errors.append(f"Expected 3-5 bullet points, got {len(bullets)}.")
    return errors


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Planner",
            contract=CONTRACT,
            determinism=DeterminismConfig(temperature=0.1, top_p=0.9, max_retries=2, timeout_s=30.0),
        )

    def _prompt(self, tx: dict) -> str:
        return f"""Incoming transaction:
- Amount: ${tx.get('amount', 0):.2f}
- Merchant type: {tx.get('merchant_type', 'unknown')}
- Location: {tx.get('location', 'unknown')}
- Hour: {tx.get('hour', 12)}:00
- Is home city: {bool(tx.get('is_home_city', 1))}
- Is known merchant: {bool(tx.get('is_known_merchant', 1))}
- Is late night: {bool(tx.get('is_late_night', 0))}
- Amount vs user average: {tx.get('amount_vs_avg', 1.0):.2f}x

What are the key risk factors that need to be investigated for this transaction?"""

    def plan(self, tx: dict) -> str:
        return self.run(self._prompt(tx), SYSTEM, output_validator=_validate_planner_output)

    def plan_stream(self, tx: dict):
        return self.run_stream(self._prompt(tx), SYSTEM, output_validator=_validate_planner_output)
