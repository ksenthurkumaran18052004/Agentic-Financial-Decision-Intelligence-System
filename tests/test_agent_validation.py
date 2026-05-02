import pytest

from src.agents.planner_agent import PlannerAgent, CONTRACT as PLANNER_CONTRACT
from src.agents.validation import validate_inputs_with_contract


def test_planner_validate_good_input():
    ag = PlannerAgent()
    tx = {
        "transaction_id": "tx-1",
        "amount": 12.34,
        "merchant_type": "retail",
        "location": "NY",
        "hour": 14,
        "is_home_city": 1,
        "is_known_merchant": 1,
        "is_late_night": 0,
        "amount_vs_avg": 1.2,
    }
    # ensure base validate_inputs is consistent with jsonschema helper
    errors_base = ag.validate_inputs(tx)
    errors_js = validate_inputs_with_contract(PLANNER_CONTRACT.input_schema, tx)
    assert errors_base == errors_js


def test_planner_validate_missing_field():
    ag = PlannerAgent()
    tx = {"amount": 12.0}
    errors = ag.validate_inputs(tx)
    assert any("Missing required input" in e or "expected" in e for e in errors)
