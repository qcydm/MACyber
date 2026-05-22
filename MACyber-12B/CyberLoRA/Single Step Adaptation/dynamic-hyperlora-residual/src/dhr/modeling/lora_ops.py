from __future__ import annotations

from collections.abc import Mapping
from numbers import Number
from typing import Any

import torch


TensorMap = Mapping[str, torch.Tensor]


def _normalize_payload(payload: Any, arg_name: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        return {"default_layer": payload}

    if isinstance(payload, Mapping):
        normalized: dict[str, torch.Tensor] = {}
        for layer_name, value in payload.items():
            if not isinstance(layer_name, str):
                raise TypeError(f"{arg_name} keys must be str, got: {type(layer_name)!r}")
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{arg_name}[{layer_name!r}] must be torch.Tensor, got: {type(value)!r}"
                )
            normalized[layer_name] = value
        return normalized

    raise TypeError(
        f"{arg_name} must be a torch.Tensor or Mapping[str, torch.Tensor], "
        f"got: {type(payload)!r}"
    )


def _ensure_same_keys(reference: TensorMap, candidate: TensorMap, candidate_name: str) -> None:
    ref_keys = set(reference.keys())
    candidate_keys = set(candidate.keys())
    if ref_keys != candidate_keys:
        raise ValueError(
            f"Layer keys mismatch for {candidate_name}: expected {sorted(ref_keys)}, "
            f"got {sorted(candidate_keys)}"
        )


def _validate_factor_pair(a_factor: torch.Tensor, b_factor: torch.Tensor, layer_name: str) -> None:
    if a_factor.dim() not in {2, 3}:
        raise ValueError(
            f"{layer_name}: A factor must be 2D/3D tensor, got dim={a_factor.dim()}"
        )
    if b_factor.dim() != a_factor.dim():
        raise ValueError(
            f"{layer_name}: A/B rank dims mismatch, got A dim={a_factor.dim()}, "
            f"B dim={b_factor.dim()}"
        )

    if a_factor.dim() == 2:
        if a_factor.shape[1] != b_factor.shape[0]:
            raise ValueError(
                f"{layer_name}: invalid 2D LoRA factors, A shape={tuple(a_factor.shape)}, "
                f"B shape={tuple(b_factor.shape)}"
            )
        return

    # batched factors
    if a_factor.shape[0] != b_factor.shape[0]:
        raise ValueError(
            f"{layer_name}: batch size mismatch for 3D factors, "
            f"A batch={a_factor.shape[0]}, B batch={b_factor.shape[0]}"
        )
    if a_factor.shape[2] != b_factor.shape[1]:
        raise ValueError(
            f"{layer_name}: invalid 3D LoRA factors, A shape={tuple(a_factor.shape)}, "
            f"B shape={tuple(b_factor.shape)}"
        )


def _to_alpha_tensor(alpha_value: Any, layer_name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(alpha_value, torch.Tensor):
        alpha_tensor = alpha_value.to(device=device, dtype=dtype)
    elif isinstance(alpha_value, Number):
        alpha_tensor = torch.tensor(alpha_value, device=device, dtype=dtype)
    else:
        raise TypeError(
            f"alpha[{layer_name!r}] must be a scalar number or torch.Tensor, "
            f"got: {type(alpha_value)!r}"
        )

    if alpha_tensor.dim() == 2 and alpha_tensor.shape[-1] == 1:
        alpha_tensor = alpha_tensor.squeeze(-1)

    if alpha_tensor.dim() > 1:
        raise ValueError(
            f"alpha[{layer_name!r}] must be scalar or 1D tensor, "
            f"got shape={tuple(alpha_tensor.shape)}"
        )
    return alpha_tensor


def compute_lora_weight(a_factor: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
    """Compute LoRA weight matrix (or batch of matrices): A @ B."""
    _validate_factor_pair(a_factor, b_factor, "default_layer")
    if a_factor.dim() == 2:
        return a_factor @ b_factor
    return torch.bmm(a_factor, b_factor)


def compose_delta(
    a0: Any,
    b0: Any,
    a1: Any,
    b1: Any,
    alpha: Any,
) -> dict[str, torch.Tensor]:
    """
    Compose DeltaW = A0B0 + alpha * (A1B1) without cross terms.

    Supports:
    - per-layer tensors via Mapping[str, Tensor]
    - single-layer tensors (treated as key: "default_layer")
    - alpha as global scalar/tensor or per-layer mapping
    """
    a0_map = _normalize_payload(a0, "a0")
    b0_map = _normalize_payload(b0, "b0")
    a1_map = _normalize_payload(a1, "a1")
    b1_map = _normalize_payload(b1, "b1")

    _ensure_same_keys(a0_map, b0_map, "b0")
    _ensure_same_keys(a0_map, a1_map, "a1")
    _ensure_same_keys(a0_map, b1_map, "b1")

    alpha_map: dict[str, Any]
    if isinstance(alpha, Mapping):
        alpha_map = dict(alpha)
    else:
        alpha_map = {layer_name: alpha for layer_name in a0_map}

    _ensure_same_keys(a0_map, {k: torch.tensor(0) for k in alpha_map}, "alpha")

    delta_per_layer: dict[str, torch.Tensor] = {}
    for layer_name in a0_map:
        a0_factor = a0_map[layer_name]
        b0_factor = b0_map[layer_name]
        a1_factor = a1_map[layer_name]
        b1_factor = b1_map[layer_name]

        _validate_factor_pair(a0_factor, b0_factor, layer_name)
        _validate_factor_pair(a1_factor, b1_factor, layer_name)

        base_weight = compute_lora_weight(a0_factor, b0_factor)
        residual_weight = compute_lora_weight(a1_factor, b1_factor)
        if base_weight.shape[-2:] != residual_weight.shape[-2:]:
            raise ValueError(
                f"{layer_name}: base/residual projected shape mismatch, "
                f"base={tuple(base_weight.shape)}, residual={tuple(residual_weight.shape)}"
            )

        alpha_tensor = _to_alpha_tensor(
            alpha_map[layer_name],
            layer_name,
            dtype=base_weight.dtype,
            device=base_weight.device,
        )

        if alpha_tensor.dim() == 0:
            delta_weight = base_weight + alpha_tensor * residual_weight
            delta_per_layer[layer_name] = delta_weight
            continue

        # alpha is 1D batch vector
        batch_size = alpha_tensor.shape[0]
        if base_weight.dim() == 2:
            base_weight = base_weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if base_weight.shape[0] != batch_size:
                raise ValueError(
                    f"{layer_name}: alpha batch={batch_size} mismatches projected "
                    f"batch={base_weight.shape[0]}"
                )

        if residual_weight.dim() == 2:
            residual_weight = residual_weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if residual_weight.shape[0] != batch_size:
                raise ValueError(
                    f"{layer_name}: alpha batch={batch_size} mismatches residual "
                    f"batch={residual_weight.shape[0]}"
                )

        delta_weight = base_weight + alpha_tensor.view(batch_size, 1, 1) * residual_weight
        delta_per_layer[layer_name] = delta_weight

    return delta_per_layer

