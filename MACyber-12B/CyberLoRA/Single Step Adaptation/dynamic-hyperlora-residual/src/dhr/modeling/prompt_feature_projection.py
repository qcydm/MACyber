from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


PROMPT_FEATURE_SCHEMA_VERSION = 1
PROMPT_FEATURE_PACK_COMPONENTS = ("mean", "last", "max")


def resolve_prompt_feature_hidden_size(base_model: Any, fallback_hidden_size: int) -> int:
    if hasattr(base_model, "config") and hasattr(base_model.config, "hidden_size"):
        return int(base_model.config.hidden_size)
    return int(fallback_hidden_size)


def prompt_feature_proj_in_dim(hidden_size: int) -> int:
    return int(hidden_size) * len(PROMPT_FEATURE_PACK_COMPONENTS)


def build_prompt_feature_projection(
    hidden_size: int,
    feature_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> nn.Linear:
    proj = nn.Linear(prompt_feature_proj_in_dim(hidden_size), feature_dim, bias=False)
    proj.to(device=device, dtype=dtype)
    return proj


def prompt_feature_projection_metadata(hidden_size: int) -> dict[str, Any]:
    return {
        "prompt_feature_schema_version": PROMPT_FEATURE_SCHEMA_VERSION,
        "prompt_feature_pack_components": list(PROMPT_FEATURE_PACK_COMPONENTS),
        "prompt_feature_proj_in_dim": prompt_feature_proj_in_dim(hidden_size),
    }


def _checkpoint_label(checkpoint_path: str | Path | None) -> str:
    if checkpoint_path is None:
        return "<in-memory checkpoint>"
    return str(Path(checkpoint_path))


def load_prompt_feature_projection(
    checkpoint_state: Mapping[str, Any],
    *,
    hidden_size: int,
    feature_dim: int,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    required: bool,
) -> nn.Linear | None:
    ckpt_label = _checkpoint_label(checkpoint_path)
    state_dict = checkpoint_state.get("prompt_feature_proj")
    if state_dict is None:
        if required:
            raise ValueError(
                f"Checkpoint missing 'prompt_feature_proj' for prompt-derived features: {ckpt_label}"
            )
        return None

    expected_metadata = prompt_feature_projection_metadata(hidden_size)
    missing_meta = [key for key in expected_metadata if key not in checkpoint_state]
    if missing_meta:
        if required:
            raise ValueError(
                "Checkpoint missing prompt feature projection metadata "
                f"{missing_meta} for new schema: {ckpt_label}"
            )
        return None

    if int(checkpoint_state["prompt_feature_schema_version"]) != PROMPT_FEATURE_SCHEMA_VERSION:
        if required:
            raise ValueError(
                "Prompt feature schema version mismatch: "
                f"expected={PROMPT_FEATURE_SCHEMA_VERSION} "
                f"got={checkpoint_state['prompt_feature_schema_version']} "
                f"checkpoint={ckpt_label}"
            )
        return None

    actual_components = list(checkpoint_state["prompt_feature_pack_components"])
    expected_components = list(PROMPT_FEATURE_PACK_COMPONENTS)
    if actual_components != expected_components:
        if required:
            raise ValueError(
                "Prompt feature pack components mismatch: "
                f"expected={expected_components} got={actual_components} checkpoint={ckpt_label}"
            )
        return None

    expected_in_dim = prompt_feature_proj_in_dim(hidden_size)
    actual_in_dim = int(checkpoint_state["prompt_feature_proj_in_dim"])
    if actual_in_dim != expected_in_dim:
        if required:
            raise ValueError(
                "Prompt feature projection input dim mismatch: "
                f"expected={expected_in_dim} got={actual_in_dim} checkpoint={ckpt_label}"
            )
        return None

    weight = state_dict.get("weight") if isinstance(state_dict, Mapping) else None
    expected_weight_shape = (feature_dim, expected_in_dim)
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected_weight_shape:
        if required:
            actual_shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else None
            raise ValueError(
                "prompt_feature_proj weight shape mismatch: "
                f"expected={expected_weight_shape} got={actual_shape} checkpoint={ckpt_label}"
            )
        return None

    proj = build_prompt_feature_projection(
        hidden_size=hidden_size,
        feature_dim=feature_dim,
        device=device,
        dtype=torch.float32,
    )
    proj.load_state_dict(dict(state_dict), strict=True)
    proj.eval()
    return proj
