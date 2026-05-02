"""
Base agent — wraps Together AI (OpenAI-compatible) for all agents.
"""

import sys
from dataclasses import asdict, dataclass, field
from time import perf_counter
from pathlib import Path
import json
from typing import Callable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    LLM_SMART,
    AGENT_TEMPERATURE,
    AGENT_TOP_P,
    AGENT_MAX_RETRIES,
    AGENT_TIMEOUT_S,
)


OutputValidator = Callable[[str], list[str]]


@dataclass(frozen=True)
class AgentContract:
    agent_name: str
    version: str
    schema_version: str
    role: str
    objective: str
    instructions: str
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    input_schema: dict[str, str] = field(default_factory=dict)
    output_schema: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterminismConfig:
    temperature: float = AGENT_TEMPERATURE
    top_p: float = AGENT_TOP_P
    max_retries: int = AGENT_MAX_RETRIES
    timeout_s: float = AGENT_TIMEOUT_S


@dataclass
class AgentRunTrace:
    agent_name: str
    contract_version: str
    schema_version: str
    model: str
    latency_ms: int
    retries: int
    output_valid: bool
    validation_errors: list[str] = field(default_factory=list)


class AgentExecutionError(RuntimeError):
    """Raised when transient API failures persist beyond retry budget."""


class BaseAgent:
    def __init__(
        self,
        model: str = None,
        name: str = "Agent",
        contract: AgentContract | None = None,
        determinism: DeterminismConfig | None = None,
    ):
        self.name = name
        self.model = model or LLM_SMART
        self.contract = contract or AgentContract(
            agent_name=name.lower(),
            version="1.0.0",
            schema_version="1.0.0",
            role=f"{name} specialist",
            objective="Produce domain-grounded analysis for the orchestrator.",
            instructions="Use only provided context and be explicit when uncertain.",
        )
        self.determinism = determinism or DeterminismConfig()
        self.last_trace: AgentRunTrace | None = None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
        return self._client

    def _contract_system_prompt(self) -> str:
        return f"""You are a governed agent in a financial decision intelligence pipeline.

Agent Contract:
- Agent: {self.contract.agent_name}
- Contract Version: {self.contract.version}
- Schema Version: {self.contract.schema_version}
- Role: {self.contract.role}
- Objective: {self.contract.objective}

Instructions:
{self.contract.instructions}

Skills:
{json.dumps(list(self.contract.skills), ensure_ascii=True)}

Tools:
{json.dumps(list(self.contract.tools), ensure_ascii=True)}

Constraints:
{json.dumps(list(self.contract.constraints), ensure_ascii=True)}

Input Schema:
{json.dumps(self.contract.input_schema, indent=2, sort_keys=True, ensure_ascii=True)}

Output Schema:
{json.dumps(self.contract.output_schema, indent=2, sort_keys=True, ensure_ascii=True)}

If data is incomplete or ambiguous, explicitly state uncertainty.
Do not infer facts not present in the input."""

    def _messages(self, user_prompt: str, system_prompt: str = None) -> list:
        msgs = [{"role": "system", "content": self._contract_system_prompt()}]
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def _create_completion(self, messages: list, max_tokens: int, stream: bool):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        for attempt in range(self.determinism.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.determinism.temperature,
                    top_p=self.determinism.top_p,
                    max_tokens=max_tokens,
                    timeout=self.determinism.timeout_s,
                    stream=stream,
                )
                return response, attempt
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as exc:
                if attempt == self.determinism.max_retries:
                    raise AgentExecutionError(
                        f"{self.name} failed after {self.determinism.max_retries + 1} attempts."
                    ) from exc

        raise AgentExecutionError(f"{self.name} failed before producing a response.")

    def _apply_validation(self, raw_output: str, output_validator: OutputValidator | None):
        output = raw_output.strip()
        errors: list[str] = []

        if not output:
            errors.append("Model returned empty output.")
        if output_validator:
            errors.extend(output_validator(output))

        if errors:
            output = (
                f"UNCERTAIN_OUTPUT: {self.name} could not produce schema-compliant output. "
                f"Errors: {'; '.join(errors)}."
            )

        return output, len(errors) == 0, errors

    def run(
        self,
        user_prompt: str,
        system_prompt: str = None,
        max_tokens: int = 600,
        output_validator: OutputValidator | None = None,
    ) -> str:
        start = perf_counter()
        response, retries = self._create_completion(
            messages=self._messages(user_prompt, system_prompt),
            max_tokens=max_tokens,
            stream=False,
        )
        raw_output = response.choices[0].message.content or ""
        output, output_valid, validation_errors = self._apply_validation(raw_output, output_validator)
        latency_ms = int((perf_counter() - start) * 1000)
        self.last_trace = AgentRunTrace(
            agent_name=self.contract.agent_name,
            contract_version=self.contract.version,
            schema_version=self.contract.schema_version,
            model=self.model,
            latency_ms=latency_ms,
            retries=retries,
            output_valid=output_valid,
            validation_errors=validation_errors,
        )
        return output

    def run_stream(
        self,
        user_prompt: str,
        system_prompt: str = None,
        max_tokens: int = 600,
        output_validator: OutputValidator | None = None,
    ):
        """Yield tokens one by one — use with st.write_stream()."""
        start = perf_counter()
        stream, retries = self._create_completion(
            messages=self._messages(user_prompt, system_prompt),
            max_tokens=max_tokens,
            stream=True,
        )

        buffered: list[str] = []
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                buffered.append(token)
                yield token

        output, output_valid, validation_errors = self._apply_validation("".join(buffered), output_validator)
        latency_ms = int((perf_counter() - start) * 1000)
        self.last_trace = AgentRunTrace(
            agent_name=self.contract.agent_name,
            contract_version=self.contract.version,
            schema_version=self.contract.schema_version,
            model=self.model,
            latency_ms=latency_ms,
            retries=retries,
            output_valid=output_valid,
            validation_errors=validation_errors,
        )

        if not output_valid:
            yield f"\n\n{output}"

    def contract_metadata(self) -> dict:
        return {
            "agent_name": self.contract.agent_name,
            "version": self.contract.version,
            "schema_version": self.contract.schema_version,
            "model": self.model,
        }

    def validate_inputs(self, inputs: dict) -> list[str]:
        """
        Basic runtime validation of a provided inputs dict against the agent's
        declared `contract.input_schema`.

        Returns a list of error messages. Empty list means validation passed.
        This is intentionally lightweight: it checks presence and basic types
        (string, float, int, dict, list, bool/0|1) so the orchestrator can
        conservatively avoid running agents on malformed input.
        """
        errors: list[str] = []
        schema = self.contract.input_schema or {}
        for key, expected in schema.items():
            if key not in inputs:
                errors.append(f"Missing required input '{key}'")
                continue
            val = inputs[key]
            exp = str(expected).lower()
            if "float" in exp:
                if not isinstance(val, (float, int)):
                    errors.append(f"Input '{key}' expected float, got {type(val).__name__}")
            elif "int" in exp or "integer" in exp:
                if not isinstance(val, int):
                    errors.append(f"Input '{key}' expected int, got {type(val).__name__}")
            elif "0|1" in exp:
                if val not in (0, 1, True, False):
                    errors.append(f"Input '{key}' expected 0|1 or boolean, got {val}")
            elif "string" in exp or "str" in exp:
                if not isinstance(val, str):
                    errors.append(f"Input '{key}' expected string, got {type(val).__name__}")
            elif "dict" in exp:
                if not isinstance(val, dict):
                    errors.append(f"Input '{key}' expected dict, got {type(val).__name__}")
            elif "list" in exp:
                if not isinstance(val, (list, tuple)):
                    errors.append(f"Input '{key}' expected list, got {type(val).__name__}")
            # otherwise allow lenient checking for other descriptors

        return errors

    def latest_evaluation(self) -> dict | None:
        if self.last_trace is None:
            return None
        return asdict(self.last_trace)
