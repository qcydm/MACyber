from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _as_layer_map(payload: Any, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be Mapping[str, torch.Tensor], got {type(payload)!r}")
    output: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be str, got {type(key)!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}[{key!r}] must be torch.Tensor, got {type(value)!r}")
        output[key] = value
    return output


def _orthogonal_regularization(a_factor: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
    a_factor = torch.nan_to_num(a_factor, nan=0.0, posinf=1e4, neginf=-1e4)
    b_factor = torch.nan_to_num(b_factor, nan=0.0, posinf=1e4, neginf=-1e4)
    if a_factor.dim() == 2:
        rank = a_factor.shape[1]
        eye = torch.eye(rank, device=a_factor.device, dtype=a_factor.dtype)
        a_cov = a_factor.transpose(0, 1) @ a_factor
        b_cov = b_factor @ b_factor.transpose(0, 1)
        return ((a_cov - eye) ** 2).mean() + ((b_cov - eye) ** 2).mean()

    # Batched factors.
    rank = a_factor.shape[2]
    eye = torch.eye(rank, device=a_factor.device, dtype=a_factor.dtype).unsqueeze(0)
    a_cov = torch.matmul(a_factor.transpose(1, 2), a_factor)
    b_cov = torch.matmul(b_factor, b_factor.transpose(1, 2))
    return ((a_cov - eye) ** 2).mean() + ((b_cov - eye) ** 2).mean()


def _linalg_safe_input(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.dtype in {torch.float32, torch.float64}:
        return matrix
    return matrix.to(dtype=torch.float32)


def _svdvals_from_low_rank_factors(a_factor: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
    """
    Exact singular values of W=A@B via a small core matrix.

    Let A=Q_a R_a and B^T=Q_b R_b (reduced QR), then:
        W = A B = Q_a (R_a R_b^T) Q_b^T
    so singular values(W) == singular values(R_a R_b^T).
    """
    autocast_guard = (
        torch.autocast(device_type="cuda", enabled=False)
        if a_factor.device.type == "cuda"
        else nullcontext()
    )
    with autocast_guard:
        a_safe = _linalg_safe_input(a_factor)
        b_safe = _linalg_safe_input(b_factor)
        _, r_a = torch.linalg.qr(a_safe, mode="reduced")
        _, r_bt = torch.linalg.qr(b_safe.transpose(-2, -1), mode="reduced")
        core = _linalg_safe_input(r_a @ r_bt.transpose(-2, -1))
        return torch.linalg.svdvals(core)


def _rank_regularization(a_factor: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
    a_factor = torch.nan_to_num(a_factor, nan=0.0, posinf=1e4, neginf=-1e4)
    b_factor = torch.nan_to_num(b_factor, nan=0.0, posinf=1e4, neginf=-1e4)
    # The initial residual LoRA path uses zero-initialized B factors so the injected
    # residual weight starts at zero. `torch.linalg.svdvals` can emit NaN gradients for
    # an exactly-zero matrix, which then corrupts the first optimizer step. In that cold
    # start case the rank penalty should be exactly zero with zero gradient.
    if torch.count_nonzero(a_factor).item() == 0 or torch.count_nonzero(b_factor).item() == 0:
        return torch.zeros((), device=a_factor.device, dtype=a_factor.dtype)
    singular_values = _svdvals_from_low_rank_factors(a_factor, b_factor)
    return singular_values.abs().mean()


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    residual_factors: Mapping[str, Any],
    lambda_orth: float = 1e-3,
    lambda_rank: float = 5e-4,
) -> dict[str, torch.Tensor]:
    """
    Compute L = L_task + lambda_orth * L_orth + lambda_rank * L_rank.

    logits: [B, seq_len, vocab_size]  — LM token-level logits
    labels: [B, seq_len]              — token ids with -100 for masked (prompt) positions

    Returns structured loss payload: total / task / orth / rank.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be shape [B, seq_len, vocab_size], got {tuple(logits.shape)}"
        )
    if labels.dim() != 2:
        raise ValueError(
            f"labels must be shape [B, seq_len], got {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits/labels shape mismatch: {tuple(logits.shape[:2])} vs {tuple(labels.shape)}"
        )

    vocab_size = logits.shape[-1]
    # Shift: predict token i+1 from position i (standard causal LM loss)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    task_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1).to(device=shift_logits.device, dtype=torch.long),
        ignore_index=-100,
    )

    lora_a1 = _as_layer_map(residual_factors.get("lora_A1"), "residual_factors['lora_A1']")
    lora_b1 = _as_layer_map(residual_factors.get("lora_B1"), "residual_factors['lora_B1']")
    if set(lora_a1.keys()) != set(lora_b1.keys()):
        raise ValueError("residual_factors lora_A1/lora_B1 keys mismatch")

    orth_loss = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)
    rank_loss = torch.zeros((), device=task_loss.device, dtype=task_loss.dtype)

    layer_count = max(len(lora_a1), 1)
    for layer_name in lora_a1:
        a_factor = lora_a1[layer_name]
        b_factor = lora_b1[layer_name]
        if a_factor.dim() not in {2, 3} or b_factor.dim() != a_factor.dim():
            raise ValueError(
                f"{layer_name}: expected A/B to be both 2D or 3D, "
                f"got A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )
        orth_loss = orth_loss + _orthogonal_regularization(a_factor, b_factor)
        rank_loss = rank_loss + _rank_regularization(a_factor, b_factor)

    orth_loss = orth_loss / layer_count
    rank_loss = rank_loss / layer_count
    total_loss = task_loss + float(lambda_orth) * orth_loss + float(lambda_rank) * rank_loss
    return {
        "total": total_loss,
        "task": task_loss,
        "orth": orth_loss,
        "rank": rank_loss,
    }
