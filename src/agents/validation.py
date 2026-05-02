"""Helpers to produce JSON Schema from lightweight contract descriptors
and validate inputs/outputs using `jsonschema`.
"""
from typing import Any, Dict
import re
from jsonschema import validate, ValidationError


_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "float": "number",
    "int": "integer",
    "integer": "integer",
    "dict": "object",
    "list": "array",
    "bool": "boolean",
}


def _normalize_descriptor(desc: str) -> str:
    return desc.strip().lower()


def contract_input_to_jsonschema(input_schema: Dict[str, str]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: list[str] = []
    for key, desc in (input_schema or {}).items():
        d = _normalize_descriptor(str(desc))
        # detect 0|1 boolean shorthand
        if "0|1" in d or "bool" in d:
            props[key] = {"type": "boolean"}
            required.append(key)
            continue

        # detect simple numeric ranges like 'float in [0,1]'
        m = re.search(r"(float|int|integer).*\[(.*?)\]", d)
        if m:
            t = _TYPE_MAP.get(m.group(1), "string")
            props[key] = {"type": t}
            required.append(key)
            continue

        # basic mapping
        for token, js_type in _TYPE_MAP.items():
            if token in d:
                props[key] = {"type": js_type}
                required.append(key)
                break
        else:
            # fallback to allowing any type
            props[key] = {}
            required.append(key)

    schema = {"type": "object", "properties": props, "required": required}
    return schema


def validate_inputs_with_contract(input_schema: Dict[str, str], inputs: Dict[str, Any]) -> list[str]:
    """Validate `inputs` against a generated JSON Schema from `input_schema`.

    Returns a list of error messages (empty if valid).
    """
    schema = contract_input_to_jsonschema(input_schema)
    errors: list[str] = []
    try:
        validate(instance=inputs or {}, schema=schema)
    except ValidationError as exc:
        errors.append(str(exc.message))
    return errors
