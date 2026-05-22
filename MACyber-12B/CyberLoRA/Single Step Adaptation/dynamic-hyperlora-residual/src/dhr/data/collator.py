from __future__ import annotations

from typing import Any


def simple_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {"batch": batch, "batch_size": len(batch)}

