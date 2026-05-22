from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

class RankController:
    def __init__(self, target_rank: int = 8, tol: int = 2) -> None:
        self.target_rank = target_rank
        self.tol = tol

    def _prune_single(
        self,
        a_factor: torch.Tensor,
        b_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if a_factor.dim() != 2 or b_factor.dim() != 2:
            raise ValueError(
                f"Expected 2D factors, got A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )
        if a_factor.shape[1] != b_factor.shape[0]:
            raise ValueError(
                f"Rank mismatch between A/B: A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )

        a_factor = torch.nan_to_num(a_factor, nan=0.0, posinf=1e4, neginf=-1e4)
        b_factor = torch.nan_to_num(b_factor, nan=0.0, posinf=1e4, neginf=-1e4)
        out_features, rank_dim = a_factor.shape
        in_features = b_factor.shape[1]

        residual_weight = torch.nan_to_num(a_factor @ b_factor, nan=0.0, posinf=1e4, neginf=-1e4)
        try:
            u, singular_values, vh = torch.linalg.svd(residual_weight, full_matrices=False)
        except RuntimeError:
            rank_scores = torch.zeros(rank_dim, device=a_factor.device, dtype=a_factor.dtype)
            return a_factor, b_factor, rank_scores
        keep_rank = min(
            max(self.target_rank + self.tol, 0),
            singular_values.numel(),
            rank_dim,
        )

        if keep_rank > 0:
            s_keep = singular_values[:keep_rank].to(dtype=a_factor.dtype)
            u_keep = u[:, :keep_rank].to(dtype=a_factor.dtype)
            vh_keep = vh[:keep_rank, :].to(dtype=b_factor.dtype)
            sqrt_s = torch.sqrt(torch.clamp(s_keep, min=0.0))

            a_keep = u_keep * sqrt_s.unsqueeze(0)
            b_keep = sqrt_s.unsqueeze(1) * vh_keep
            a_pad = torch.zeros(
                out_features,
                rank_dim - keep_rank,
                device=a_factor.device,
                dtype=a_factor.dtype,
            )
            b_pad = torch.zeros(
                rank_dim - keep_rank,
                in_features,
                device=b_factor.device,
                dtype=b_factor.dtype,
            )
            pruned_a = torch.cat([a_keep, a_pad], dim=1)
            pruned_b = torch.cat([b_keep, b_pad], dim=0)
            score_keep = singular_values[:keep_rank].abs().to(dtype=a_factor.dtype)
            score_pad = torch.zeros(
                rank_dim - keep_rank,
                device=a_factor.device,
                dtype=a_factor.dtype,
            )
            rank_scores = torch.cat([score_keep, score_pad], dim=0)
        else:
            pruned_a = torch.zeros(
                out_features,
                rank_dim,
                device=a_factor.device,
                dtype=a_factor.dtype,
            )
            pruned_b = torch.zeros(
                rank_dim,
                in_features,
                device=b_factor.device,
                dtype=b_factor.dtype,
            )
            rank_scores = torch.zeros(rank_dim, device=a_factor.device, dtype=a_factor.dtype)
        return pruned_a, pruned_b, rank_scores

    def _prune_pair(
        self,
        a_factor: torch.Tensor,
        b_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if a_factor.dim() == 2:
            return self._prune_single(a_factor, b_factor)

        if a_factor.dim() != 3 or b_factor.dim() != 3:
            raise ValueError(
                f"Expected 2D/3D factors, got A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )
        if a_factor.shape[0] != b_factor.shape[0]:
            raise ValueError(
                f"Batched A/B first dim mismatch: A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )
        if a_factor.shape[2] != b_factor.shape[1]:
            raise ValueError(
                f"Batched rank mismatch: A={tuple(a_factor.shape)} B={tuple(b_factor.shape)}"
            )

        pruned_a_batch = []
        pruned_b_batch = []
        score_batch = []
        for idx in range(a_factor.shape[0]):
            pa, pb, rs = self._prune_single(a_factor[idx], b_factor[idx])
            pruned_a_batch.append(pa)
            pruned_b_batch.append(pb)
            score_batch.append(rs)
        return (
            torch.stack(pruned_a_batch, dim=0),
            torch.stack(pruned_b_batch, dim=0),
            torch.stack(score_batch, dim=0),
        )

    def prune(self, payload: Any) -> Any:
        if not isinstance(payload, Mapping):
            raise TypeError(f"payload must be mapping, got {type(payload)!r}")
        if "lora_A1" not in payload or "lora_B1" not in payload:
            raise ValueError("payload must include lora_A1 and lora_B1")

        lora_a1 = payload["lora_A1"]
        lora_b1 = payload["lora_B1"]
        if not isinstance(lora_a1, Mapping) or not isinstance(lora_b1, Mapping):
            raise TypeError("payload lora_A1/lora_B1 must be mapping")
        if set(lora_a1.keys()) != set(lora_b1.keys()):
            raise ValueError("payload lora_A1/lora_B1 keys mismatch")

        pruned_a: dict[str, torch.Tensor] = {}
        pruned_b: dict[str, torch.Tensor] = {}
        rank_scores: dict[str, torch.Tensor] = {}
        effective_rank: dict[str, int] = {}

        for layer_name in lora_a1:
            a_factor = lora_a1[layer_name]
            b_factor = lora_b1[layer_name]
            if not isinstance(a_factor, torch.Tensor) or not isinstance(b_factor, torch.Tensor):
                raise TypeError(f"{layer_name}: A/B factors must be torch.Tensor")
            pa, pb, rs = self._prune_pair(a_factor=a_factor, b_factor=b_factor)
            pruned_a[layer_name] = pa
            pruned_b[layer_name] = pb
            rank_scores[layer_name] = rs

            if pa.dim() == 2:
                rank_val = torch.linalg.matrix_rank((pa @ pb).float()).item()
                effective_rank[layer_name] = int(rank_val)
            else:
                rank_vals = [torch.linalg.matrix_rank((pa[i] @ pb[i]).float()).item() for i in range(pa.shape[0])]
                effective_rank[layer_name] = int(max(rank_vals) if rank_vals else 0)

        updated = dict(payload)
        updated["lora_A1"] = pruned_a
        updated["lora_B1"] = pruned_b
        updated["rank_scores"] = rank_scores
        updated["effective_rank"] = effective_rank
        return updated

