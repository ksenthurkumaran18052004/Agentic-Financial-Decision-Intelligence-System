"""Centralized agent registry.

This module imports per-agent `CONTRACT` objects and exposes a registry
mapping of agent_name -> contract and helper functions for discovery.
"""
from typing import Dict

from src.agents.planner_agent import CONTRACT as PLANNER_CONTRACT
from src.agents.risk_agent import CONTRACT as RISK_CONTRACT
from src.agents.fraud_agent import CONTRACT as FRAUD_CONTRACT
from src.agents.insight_agent import CONTRACT as INSIGHT_CONTRACT
from src.agents.evaluator_agent import CONTRACT as EVALUATOR_CONTRACT

_REGISTRY: Dict[str, object] = {
    PLANNER_CONTRACT.agent_name: PLANNER_CONTRACT,
    RISK_CONTRACT.agent_name: RISK_CONTRACT,
    FRAUD_CONTRACT.agent_name: FRAUD_CONTRACT,
    INSIGHT_CONTRACT.agent_name: INSIGHT_CONTRACT,
    EVALUATOR_CONTRACT.agent_name: EVALUATOR_CONTRACT,
}


def list_agents() -> list[str]:
    return list(_REGISTRY.keys())


def get_contract(agent_name: str):
    return _REGISTRY.get(agent_name)


def all_contracts() -> Dict[str, object]:
    return dict(_REGISTRY)
