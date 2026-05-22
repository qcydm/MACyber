from __future__ import annotations

from typing import Any

import torch


def build_prompt_text(record: dict[str, Any]) -> str:
    """Build the canonical prompt-conditioning text.

    This helper is only for HyperNet conditioning paths. It must not be reused
    for LM prompt/label-masking or generation prompt assembly.
    """

    instruction = str(record.get("instruction", "")).strip()
    inp = str(record.get("input", "")).strip()
    return f"{instruction}\n\n{inp}".strip() if inp else instruction


def pack_hidden_features(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Pack mean / last-valid / max features into a single 2D tensor."""

    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [B, T, H], got shape={tuple(hidden.shape)}")

    batch_size, seq_len, hidden_size = hidden.shape
    if seq_len == 0:
        raise ValueError("hidden must contain at least one token")

    if attention_mask is None:
        mean_pool = hidden.mean(dim=1)
        last_valid = hidden[:, -1, :]
        max_pool = hidden.max(dim=1).values
        return torch.cat([mean_pool, last_valid, max_pool], dim=-1)

    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be [B, T], got shape={tuple(attention_mask.shape)}"
        )
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            "attention_mask shape mismatch: "
            f"expected {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}"
        )

    mask_bool = attention_mask.to(device=hidden.device).bool()
    valid_counts = mask_bool.sum(dim=1)
    if torch.any(valid_counts == 0):
        bad_rows = torch.nonzero(valid_counts == 0, as_tuple=False).view(-1).tolist()
        raise ValueError(f"attention_mask contains all-pad rows with no valid tokens: {bad_rows}")

    mask_f = mask_bool.unsqueeze(-1).to(dtype=hidden.dtype)
    mean_pool = (hidden * mask_f).sum(dim=1) / valid_counts.to(dtype=hidden.dtype).unsqueeze(-1)

    positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(batch_size, -1)
    last_indices = (positions * mask_bool.long()).max(dim=1).values
    batch_indices = torch.arange(batch_size, device=hidden.device)
    last_valid = hidden[batch_indices, last_indices, :]

    neg_inf = torch.tensor(torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device)
    masked_hidden = hidden.masked_fill(~mask_bool.unsqueeze(-1), neg_inf)
    max_pool = masked_hidden.max(dim=1).values

    packed = torch.cat([mean_pool, last_valid, max_pool], dim=-1)
    if packed.shape != (batch_size, hidden_size * 3):
        raise RuntimeError(
            "packed hidden feature shape mismatch: "
            f"expected {(batch_size, hidden_size * 3)}, got {tuple(packed.shape)}"
        )
    return packed
