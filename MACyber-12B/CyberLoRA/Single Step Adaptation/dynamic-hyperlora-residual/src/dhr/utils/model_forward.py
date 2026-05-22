"""Helpers for HuggingFace model forward outputs (BaseModel vs CausalLM)."""

from __future__ import annotations

import torch
from torch import nn


def last_hidden_state_from_base_model(base_model: nn.Module, **tokenized) -> torch.Tensor:
    """Return last_hidden_state from a HF backbone.

    CausalLM wrappers (e.g. GemmaForCausalLM) expose hidden states on ``model.model(...)``.
    Plain ``AutoModel`` / BaseModel returns ``last_hidden_state`` on the top-level forward.
    """
    if hasattr(base_model, "model"):
        inner_out = base_model.model(**tokenized)
        if hasattr(inner_out, "last_hidden_state") and inner_out.last_hidden_state is not None:
            return inner_out.last_hidden_state
    out = base_model(**tokenized)
    if isinstance(out, dict):
        hidden = out.get("last_hidden_state")
    else:
        hidden = getattr(out, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    raise RuntimeError(
        f"Could not obtain last_hidden_state from base_model (type={type(base_model).__name__})"
    )
