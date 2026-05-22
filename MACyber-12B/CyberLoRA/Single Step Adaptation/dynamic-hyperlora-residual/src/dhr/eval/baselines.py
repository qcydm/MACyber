from __future__ import annotations

from collections.abc import Sequence


BASELINE_STATIC = "static_lora"
BASELINE_RESIDUAL_NO_RETRIEVER = "residual_no_retriever"
BASELINE_RESIDUAL_WITH_RETRIEVER = "residual_with_retriever"

DEFAULT_BASELINES = (
    BASELINE_STATIC,
    BASELINE_RESIDUAL_NO_RETRIEVER,
    BASELINE_RESIDUAL_WITH_RETRIEVER,
)
SUPPORTED_BASELINES = set(DEFAULT_BASELINES)


def resolve_baselines(raw_baselines: Sequence[str] | None) -> list[str]:
    if not raw_baselines:
        return list(DEFAULT_BASELINES)

    resolved: list[str] = []
    seen: set[str] = set()
    for item in raw_baselines:
        baseline = str(item).strip()
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(f"Unsupported baseline: {baseline!r}")
        if baseline in seen:
            continue
        seen.add(baseline)
        resolved.append(baseline)

    return resolved or list(DEFAULT_BASELINES)

