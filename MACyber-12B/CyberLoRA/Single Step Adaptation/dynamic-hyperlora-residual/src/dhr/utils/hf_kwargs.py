"""Optional kwargs for Hugging Face `from_pretrained` (e.g. Gemma3)."""

from __future__ import annotations

from typing import Any


def hf_extra_kwargs_from_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if bool(model_cfg.get("trust_remote_code")):
        out["trust_remote_code"] = True
    return out
