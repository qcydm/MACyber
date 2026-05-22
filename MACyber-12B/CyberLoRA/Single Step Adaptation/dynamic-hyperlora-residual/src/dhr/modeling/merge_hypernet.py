from __future__ import annotations

from collections.abc import Sequence
from math import log
from typing import Any

import torch
from torch import nn


SEGMENTED_TARGET_MODULES = (
    "early_q_proj",
    "mid_q_proj",
    "late_q_proj",
    "early_v_proj",
    "mid_v_proj",
    "late_v_proj",
)


def _safe_logit(prob: float) -> float:
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return log(prob / (1 - prob))


def _validate_segmented_layer_names(layer_names: Sequence[str]) -> list[str]:
    resolved = [str(layer_name) for layer_name in layer_names]
    expected = set(SEGMENTED_TARGET_MODULES)
    actual = set(resolved)
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"MergeHyperNet layer_names must not contain duplicates: {resolved}")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "MergeHyperNet requires the fixed six-key segmented contract "
            f"{list(SEGMENTED_TARGET_MODULES)}; missing={missing}, unexpected={unexpected}"
        )
    return resolved


class MergeHyperNet(nn.Module):
    """Predict per-layer alpha for residual merge."""

    def __init__(
        self,
        config: dict[str, Any],
        layer_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_names = _validate_segmented_layer_names(layer_names)
        if not self.layer_names:
            raise ValueError("MergeHyperNet requires non-empty layer_names")

        model_cfg = config.get("model", {})
        retriever_cfg = config.get("retriever", {})

        self.hidden_dim = int(model_cfg.get("hyper_hidden_dim", 128))
        self.stats_dim = int(retriever_cfg.get("stats_dim", 2))
        self.alpha_max = float(model_cfg.get("alpha_max", 1.5))
        self.alpha_init = float(model_cfg.get("alpha_init", 0.05))

        mlp_hidden = max(self.hidden_dim // 2, 8)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.stats_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, len(self.layer_names)),
        )

        # Keep alpha positive at cold start so the zero-initialized residual factor
        # still receives gradient on the first backward pass.
        target_prob = min(max(self.alpha_init / max(self.alpha_max, 1e-6), 1e-4), 1 - 1e-4)
        final_linear = self.mlp[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, _safe_logit(target_prob))

    @staticmethod
    def _ensure_2d_hidden(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() == 1:
            return hidden.unsqueeze(0)
        if hidden.dim() != 2:
            raise ValueError(f"hidden must be 1D/2D tensor, got shape={tuple(hidden.shape)}")
        return hidden

    def _prepare_stats(
        self,
        retriever_stats: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if retriever_stats is None:
            return torch.zeros(batch_size, self.stats_dim, device=device, dtype=dtype)

        stats = retriever_stats.to(device=device, dtype=dtype)
        if stats.dim() == 1:
            stats = stats.unsqueeze(0).expand(batch_size, -1)
        elif stats.dim() == 2:
            if stats.shape[0] != batch_size:
                raise ValueError(
                    f"retriever_stats batch mismatch: expected {batch_size}, got {stats.shape[0]}"
                )
        else:
            raise ValueError(
                f"retriever_stats must be 1D/2D tensor, got shape={tuple(stats.shape)}"
            )
        if stats.shape[1] != self.stats_dim:
            raise ValueError(
                f"retriever_stats dim mismatch: expected {self.stats_dim}, got {stats.shape[1]}"
            )
        return stats

    def forward(
        self,
        hidden: torch.Tensor,
        retriever_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_2d = self._ensure_2d_hidden(hidden)
        batch_size = hidden_2d.shape[0]
        stats = self._prepare_stats(
            retriever_stats=retriever_stats,
            batch_size=batch_size,
            dtype=hidden_2d.dtype,
            device=hidden_2d.device,
        )

        inputs = torch.cat([hidden_2d, stats], dim=-1)
        logits = self.mlp(inputs)
        alpha_matrix = self.alpha_max * torch.sigmoid(logits)

        alpha_per_layer: dict[str, torch.Tensor] = {}
        for idx, layer_name in enumerate(self.layer_names):
            layer_alpha = alpha_matrix[:, idx]
            alpha_per_layer[layer_name] = layer_alpha[0] if batch_size == 1 else layer_alpha
        return alpha_per_layer
