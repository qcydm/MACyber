from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def _validate_segmented_target_modules(
    target_modules: Sequence[str],
    *,
    source: str,
) -> list[str]:
    resolved = [str(layer_name) for layer_name in target_modules]
    expected = set(SEGMENTED_TARGET_MODULES)
    actual = set(resolved)
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{source} must not contain duplicate target_modules: {resolved}")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{source} must use the fixed six-key segmented contract {list(SEGMENTED_TARGET_MODULES)}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return resolved


@dataclass(frozen=True)
class HyperLayerShape:
    in_features: int
    out_features: int


class HyperResidualNet(nn.Module):
    """Generate residual LoRA factors A1/B1 for each target layer."""

    _A_HEAD_INIT_STD = 1.0e-3

    def __init__(
        self,
        config: dict[str, Any],
        layer_shapes: Mapping[str, HyperLayerShape | Mapping[str, int]] | None = None,
        a_head_init_std_by_layer: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        model_cfg = config.get("model", {})

        self.rank = int(model_cfg.get("residual_rank", 8))
        self.feature_dim = int(model_cfg.get("sample_feature_dim", 64))
        self.hidden_dim = int(model_cfg.get("hyper_hidden_dim", 128))
        self.num_layers = int(model_cfg.get("hyper_encoder_layers", 2))
        self.num_heads = int(model_cfg.get("hyper_num_heads", 4))

        resolved_shapes = self._resolve_layer_shapes(model_cfg, layer_shapes)
        self.layer_shapes: dict[str, HyperLayerShape] = resolved_shapes
        self.layer_names = list(resolved_shapes.keys())
        if not self.layer_names:
            raise ValueError("HyperResidualNet requires at least one target layer")
        self.a_head_init_std_by_layer = self._resolve_a_head_init_std_by_layer(
            a_head_init_std_by_layer
        )

        self.sample_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.memory_proj = nn.Linear(self.feature_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.a_heads = nn.ModuleDict()
        self.b_heads = nn.ModuleDict()
        for layer_name, shape in self.layer_shapes.items():
            self.a_heads[layer_name] = nn.Linear(self.hidden_dim, shape.out_features * self.rank)
            self.b_heads[layer_name] = nn.Linear(self.hidden_dim, shape.in_features * self.rank)
        self._init_residual_heads()

    def _init_residual_heads(self) -> None:
        """Use LoRA-style asymmetric init so A @ B starts at zero but updates immediately."""

        for layer_name in self.layer_names:
            a_head = self.a_heads[layer_name]
            b_head = self.b_heads[layer_name]
            nn.init.normal_(
                a_head.weight,
                mean=0.0,
                std=self.a_head_init_std_by_layer[layer_name],
            )
            nn.init.zeros_(a_head.bias)
            nn.init.zeros_(b_head.weight)
            nn.init.zeros_(b_head.bias)

    def _resolve_a_head_init_std_by_layer(
        self,
        a_head_init_std_by_layer: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if a_head_init_std_by_layer is None:
            return {layer_name: self._A_HEAD_INIT_STD for layer_name in self.layer_names}

        provided = {
            str(layer_name): float(std)
            for layer_name, std in a_head_init_std_by_layer.items()
        }
        missing = sorted(set(self.layer_names) - set(provided))
        unexpected = sorted(set(provided) - set(self.layer_names))
        if missing or unexpected:
            raise ValueError(
                "a_head_init_std_by_layer must match the active target layers exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for layer_name, std in provided.items():
            if not math.isfinite(std) or std < 0.0:
                raise ValueError(
                    f"a_head_init_std_by_layer[{layer_name!r}] must be a finite non-negative float, got {std!r}"
                )
        return {layer_name: provided[layer_name] for layer_name in self.layer_names}

    @staticmethod
    def _resolve_layer_shapes(
        model_cfg: dict[str, Any],
        layer_shapes: Mapping[str, HyperLayerShape | Mapping[str, int]] | None,
    ) -> dict[str, HyperLayerShape]:
        if layer_shapes is None:
            layer_shapes = {}

        target_modules = _validate_segmented_target_modules(
            model_cfg.get("target_modules", []),
            source="HyperResidualNet(model.target_modules)",
        )
        dims_cfg = model_cfg.get("target_module_dims", {})
        default_dim = int(model_cfg.get("sample_feature_dim", 64))

        unexpected_shape_keys = sorted(set(layer_shapes.keys()) - set(target_modules))
        if unexpected_shape_keys:
            raise ValueError(
                "HyperResidualNet(layer_shapes) contains keys outside the segmented contract: "
                f"{unexpected_shape_keys}"
            )

        resolved: dict[str, HyperLayerShape] = {}
        for layer_name in target_modules:
            user_shape = layer_shapes.get(layer_name)
            if isinstance(user_shape, HyperLayerShape):
                resolved[layer_name] = user_shape
                continue

            if isinstance(user_shape, Mapping):
                resolved[layer_name] = HyperLayerShape(
                    in_features=int(user_shape.get("in_features", default_dim)),
                    out_features=int(user_shape.get("out_features", default_dim)),
                )
                continue

            cfg_shape = dims_cfg.get(layer_name, {})
            resolved[layer_name] = HyperLayerShape(
                in_features=int(cfg_shape.get("in_features", default_dim)),
                out_features=int(cfg_shape.get("out_features", default_dim)),
            )
        return resolved

    @staticmethod
    def _ensure_2d_features(sample_features: torch.Tensor) -> torch.Tensor:
        if sample_features.dim() == 1:
            return sample_features.unsqueeze(0)
        if sample_features.dim() != 2:
            raise ValueError(
                f"sample_features must be 1D or 2D tensor, got shape={tuple(sample_features.shape)}"
            )
        return sample_features

    def _prepare_memory_tokens(
        self,
        memory_tokens: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if memory_tokens is None:
            return None

        memory = memory_tokens.to(device=device, dtype=dtype)
        if memory.dim() == 2:
            memory = memory.unsqueeze(0).expand(batch_size, -1, -1)
        elif memory.dim() == 3:
            if memory.shape[0] != batch_size:
                raise ValueError(
                    f"memory_tokens batch mismatch: expected {batch_size}, got {memory.shape[0]}"
                )
        else:
            raise ValueError(
                f"memory_tokens must be 2D/3D tensor, got shape={tuple(memory.shape)}"
            )
        return memory

    def forward(
        self,
        sample_features: torch.Tensor,
        memory_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        features = self._ensure_2d_features(sample_features)
        batch_size = features.shape[0]
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"sample_features dim mismatch: expected {self.feature_dim}, got {features.shape[-1]}"
            )

        features = features.to(dtype=self.sample_proj.weight.dtype, device=self.sample_proj.weight.device)
        sample_tokens = self.sample_proj(features).unsqueeze(1)

        memory = self._prepare_memory_tokens(
            memory_tokens=memory_tokens,
            batch_size=batch_size,
            dtype=features.dtype,
            device=features.device,
        )
        if memory is not None:
            if memory.shape[-1] != self.feature_dim:
                raise ValueError(
                    f"memory_tokens dim mismatch: expected {self.feature_dim}, got {memory.shape[-1]}"
                )
            memory_tokens_proj = self.memory_proj(memory)
            tokens = torch.cat([sample_tokens, memory_tokens_proj], dim=1)
        else:
            tokens = sample_tokens

        encoded = self.encoder(tokens)
        hidden = self.norm(encoded.mean(dim=1))

        lora_a1: dict[str, torch.Tensor] = {}
        lora_b1: dict[str, torch.Tensor] = {}
        for layer_name, shape in self.layer_shapes.items():
            a_flat = self.a_heads[layer_name](hidden)
            b_flat = self.b_heads[layer_name](hidden)
            a_factor = a_flat.view(batch_size, shape.out_features, self.rank)
            b_factor = b_flat.view(batch_size, self.rank, shape.in_features)

            if batch_size == 1:
                lora_a1[layer_name] = a_factor[0]
                lora_b1[layer_name] = b_factor[0]
            else:
                lora_a1[layer_name] = a_factor
                lora_b1[layer_name] = b_factor

        return {
            "lora_A1": lora_a1,
            "lora_B1": lora_b1,
            "hidden": hidden if batch_size > 1 else hidden[0],
        }
