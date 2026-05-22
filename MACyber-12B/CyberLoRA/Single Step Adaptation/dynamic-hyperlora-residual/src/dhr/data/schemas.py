from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

# Valid categorical values for generative output fields
VALID_ACTIONS = {"block", "monitor", "none"}
VALID_SEVERITIES = {"high", "medium", "low", "benign", "suspicious"}

# Regex to extract a fenced JSON block from the model output
_FENCED_JSON_BLOCK_RE = re.compile(r"```json\s*([\[{].*?[\]}])\s*```", re.DOTALL | re.IGNORECASE)

# action == "monitor" with these severities maps to label=1
_MONITOR_ATTACK_SEVERITIES = {"high", "medium"}


def _coerce_output_payload(data: Any) -> dict[str, Any] | None:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def parse_output_json(output_text: str) -> dict[str, Any] | None:
    """Extract prediction JSON from output and return a dict (or first dict item) when possible."""
    match = _FENCED_JSON_BLOCK_RE.search(output_text)
    candidate = match.group(1) if match else None

    # Fallback: accept bare JSON output when the whole text is a JSON array/object.
    stripped = output_text.strip()
    if candidate is None and (
        (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("{") and stripped.endswith("}"))
    ):
        candidate = stripped

    if candidate is None:
        return None

    try:
        data = json.loads(candidate)
        payload = _coerce_output_payload(data)
        if payload is not None:
            return payload
    except json.JSONDecodeError:
        pass
    return None


def label_from_output(output_text: str) -> int:
    """Derive 0/1 label from the JSON block inside the output text."""
    parsed = parse_output_json(output_text)
    if parsed is None:
        return 0
    action = str(parsed.get("action", "none")).lower().strip()
    severity = str(parsed.get("severity", "low")).lower().strip()
    if action == "block":
        return 1
    if action == "none":
        return 0
    # action == "monitor"
    return 1 if severity in _MONITOR_ATTACK_SEVERITIES else 0


@dataclass
class GenerativeSample:
    sample_id: str
    attack_family: str
    split: str
    instruction: str
    input: str
    output: str
    label: int  # derived from output JSON; used only for evaluation

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GenerativeSample":
        sample_id = str(payload["sample_id"])
        attack_family = str(payload["attack_family"])
        split = str(payload["split"])
        instruction = str(payload["instruction"])
        inp = str(payload["input"])
        output = str(payload["output"])
        label = int(payload.get("label", label_from_output(output)))
        if label not in {0, 1}:
            raise ValueError(f"label must be 0 or 1, got {label}")
        return cls(
            sample_id=sample_id,
            attack_family=attack_family,
            split=split,
            instruction=instruction,
            input=inp,
            output=output,
            label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
