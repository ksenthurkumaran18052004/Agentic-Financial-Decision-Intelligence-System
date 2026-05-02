Agent Design Specification
==========================

Purpose
-------
This document captures the canonical agent design used across this codebase. It describes the contract, deterministic execution, schemas, constraints, failure handling, and orchestration patterns used by agents in the multi-agent financial decision intelligence system.

Key Concepts
------------
- Agent Contract: use the `AgentContract` dataclass in `base_agent.py` to declare `agent_name`, `version`, `schema_version`, `role`, `objective`, `instructions`, `skills`, `tools`, `constraints`, `input_schema`, and `output_schema`.

- Determinism: configure `DeterminismConfig` per agent (temperature, top_p, max_retries, timeout_s) to prioritise reproducible behaviour.

- Inputs & Outputs: inputs must be structured and validated against `input_schema`. Outputs must be validated via an `output_validator` function passed to `BaseAgent.run()`; agents should return machine-parsable, schema-compliant outputs (JSON or strict text templates).

- Statelessness: agents should not keep execution-time memory between runs. All context must be passed as inputs or retrieved explicitly from external stores.

- Explainability: agents populate `AgentRunTrace` (accessible via `latest_evaluation()`) that records latency, retries, model, and validation status. Agents should produce concise human-readable reasoning as part of their outputs.

- Failure Handling: if inputs are missing or invalid, the orchestrator should not let an agent make an unconstrained decision. Instead, the orchestrator should: 1) mark agent output as `UNCERTAIN_OUTPUT`, 2) attach validation errors in the agent trace, and 3) route the pipeline toward conservative fallbacks (e.g., human review or FLAG).

How to add a new agent
----------------------
1. Create an `AgentContract` instance named `CONTRACT` in the agent module and declare `input_schema` and `output_schema` precisely.
2. Implement simple `output_validator(output: str) -> list[str]` that returns an empty list on success or a list of error messages.
3. Subclass `BaseAgent`, set `contract=CONTRACT` and `determinism=DeterminismConfig(...)` in `__init__`.
4. Have clear agent entry methods (e.g., `plan()`, `interpret()`, `analyze()`, `synthesize()`, `decide()`) that: (a) prepare structured inputs, (b) call `self.run(..., output_validator=...)`, and (c) return structured outputs (or an `UNCERTAIN_OUTPUT` sentinel when invalid).
5. Orchestrator: call `agent.validate_inputs(inputs_dict)` before invoking the agent. If errors are returned, the orchestrator should skip invoking the agent and annotate the agent trace accordingly.

Notes
-----
- Keep prompts short and factual; include the `AgentContract` system prompt (the `BaseAgent` helper does this automatically).
- Prefer explicit, machine-parseable output formats for downstream consumption.
- Use low temperature and controlled top_p for decision-critical agents.

Example
-------
See `src/agents/planner_agent.py`, `risk_agent.py`, `fraud_agent.py`, `insight_agent.py`, and `evaluator_agent.py` for working examples of contract declarations and validators.

This file is the authoritative integration guidance for agents in this repository.
