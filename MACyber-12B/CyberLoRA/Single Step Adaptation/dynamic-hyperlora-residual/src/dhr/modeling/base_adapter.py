from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dhr.modeling.lora_ops import compose_delta
from dhr.utils.hf_kwargs import hf_extra_kwargs_from_model_cfg


@dataclass(frozen=True)
class LayerSpec:
    in_features: int
    out_features: int


_SEGMENT_NAMES = ("early", "mid", "late")
_TARGET_FAMILIES = ("q_proj", "v_proj")
_EXPECTED_SEGMENTED_TARGET_MODULES = tuple(
    f"{segment}_{family}"
    for family in _TARGET_FAMILIES
    for segment in _SEGMENT_NAMES
)
_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class MockBackbone(nn.Module):
    """Small backbone used for deterministic M2 dry-run and tests."""

    def __init__(self, layer_specs: Mapping[str, LayerSpec]) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                layer_name: nn.Linear(spec.in_features, spec.out_features, bias=False)
                for layer_name, spec in layer_specs.items()
            }
        )

    def forward(self, sample_features: torch.Tensor) -> dict[str, torch.Tensor]:
        layer_outputs: dict[str, torch.Tensor] = {}
        per_layer_scores: list[torch.Tensor] = []
        for layer_name, linear in self.layers.items():
            layer_out = linear(sample_features)
            layer_outputs[layer_name] = layer_out
            per_layer_scores.append(layer_out.mean(dim=-1))

        logits = torch.stack(per_layer_scores, dim=-1).mean(dim=-1, keepdim=True)
        return {"logits": logits, "layer_outputs": layer_outputs}


class BaseAdapterManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_cfg = config.get("model", {})
        self.target_modules: list[str] = list(self.model_cfg.get("target_modules", []))
        self._validate_target_module_contract()
        self.peft_key_alignment = str(self.model_cfg.get("peft_key_alignment", "per_module")).lower()
        self.device = torch.device(self.model_cfg.get("device", "cpu"))

        self.base_model: nn.Module | None = None
        self.frozen = False
        self.is_mock_base = False

        self.layer_specs: dict[str, LayerSpec] = self._load_layer_specs_from_config()
        self._target_linear_modules: dict[str, list[nn.Linear]] = {}
        self._target_group_by_module: dict[str, str] = {}
        self._module_specs: dict[str, LayerSpec] = {}
        self._base_lora_factors: dict[str, dict[str, torch.Tensor]] = {}
        self._injection_handles: list[Any] = []
        self._latest_delta_weight: dict[str, torch.Tensor] = {}
        self._base_model_input_device = self.device
        self._is_model_sharded = False

    def _validate_target_module_contract(self) -> None:
        configured = list(self.target_modules)
        configured_set = set(configured)
        expected_set = set(_EXPECTED_SEGMENTED_TARGET_MODULES)
        if len(configured) != len(expected_set) or configured_set != expected_set:
            raise ValueError(
                "model.target_modules must use the segmented six-key contract exactly: "
                f"{list(_EXPECTED_SEGMENTED_TARGET_MODULES)}; got={configured}"
            )

    @staticmethod
    def _parse_segmented_target_key(layer_name: str) -> tuple[str, str] | None:
        for family in _TARGET_FAMILIES:
            suffix = f"_{family}"
            if layer_name.endswith(suffix):
                segment = layer_name[: -len(suffix)]
                if segment in _SEGMENT_NAMES:
                    return segment, family
        return None

    @classmethod
    def _target_family_from_module_name(cls, module_name: str) -> str | None:
        segmented = cls._parse_segmented_target_key(module_name)
        if segmented is not None:
            return segmented[1]
        for family in _TARGET_FAMILIES:
            if module_name == family or module_name.endswith(f".{family}"):
                return family
        return None

    @staticmethod
    def _extract_layer_index(module_name: str) -> int | None:
        match = _LAYER_INDEX_PATTERN.search(module_name)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _build_segment_map(layer_indices: list[int]) -> dict[int, str]:
        unique_sorted = sorted(set(layer_indices))
        if not unique_sorted:
            return {}

        total = len(unique_sorted)
        early_count = (total + 2) // 3
        mid_count = (total + 1) // 3

        segment_map: dict[int, str] = {}
        for position, layer_idx in enumerate(unique_sorted):
            if position < early_count:
                segment_map[layer_idx] = "early"
            elif position < early_count + mid_count:
                segment_map[layer_idx] = "mid"
            else:
                segment_map[layer_idx] = "late"
        return segment_map

    @staticmethod
    def _build_segmented_group_key(segment: str, family: str) -> str:
        group_key = f"{segment}_{family}"
        if group_key not in _EXPECTED_SEGMENTED_TARGET_MODULES:
            raise ValueError(
                "Unsupported segmented target module key parts: "
                f"segment={segment!r}, family={family!r}"
            )
        return group_key

    def _load_layer_specs_from_config(self) -> dict[str, LayerSpec]:
        dims_cfg = self.model_cfg.get("target_module_dims", {})
        layer_specs: dict[str, LayerSpec] = {}
        default_dim = int(self.model_cfg.get("sample_feature_dim", 64))
        for layer_name in self.target_modules:
            layer_dim_cfg = dims_cfg.get(layer_name, {})
            in_features = int(layer_dim_cfg.get("in_features", default_dim))
            out_features = int(layer_dim_cfg.get("out_features", default_dim))
            layer_specs[layer_name] = LayerSpec(in_features=in_features, out_features=out_features)
        return layer_specs

    def _build_mock_backbone(self) -> nn.Module:
        self.is_mock_base = True
        self._is_model_sharded = False
        self._base_model_input_device = self.device
        return MockBackbone(self.layer_specs).to(self.device)

    def _resolve_torch_dtype(self) -> torch.dtype | str | None:
        raw_dtype = self.model_cfg.get("base_model_dtype")
        if raw_dtype is None:
            return None
        if isinstance(raw_dtype, torch.dtype):
            return raw_dtype

        dtype_text = str(raw_dtype).strip().lower()
        if dtype_text in {"", "none", "null"}:
            return None
        if dtype_text == "auto":
            return "auto"

        alias_map: dict[str, torch.dtype] = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = alias_map.get(dtype_text)
        if resolved is None:
            raise ValueError(
                "Unsupported model.base_model_dtype. "
                "Expected one of: auto, bf16, fp16, fp32, bfloat16, float16, float32"
            )
        return resolved

    def _resolve_device_map(self) -> Any | None:
        raw_device_map = self.model_cfg.get("device_map")
        if raw_device_map is None:
            return None
        if isinstance(raw_device_map, bool):
            return "auto" if raw_device_map else None
        if isinstance(raw_device_map, Mapping):
            return dict(raw_device_map)

        device_map_text = str(raw_device_map).strip()
        if device_map_text.lower() in {"", "none", "null", "false"}:
            return None
        return device_map_text

    def _resolve_max_memory(self) -> dict[Any, str] | None:
        raw_max_memory = self.model_cfg.get("max_memory")
        if raw_max_memory is None:
            return None
        if not isinstance(raw_max_memory, Mapping):
            raise TypeError(
                f"model.max_memory must be a mapping, got: {type(raw_max_memory)!r}"
            )

        resolved: dict[Any, str] = {}
        for key, value in raw_max_memory.items():
            normalized_key: Any = key
            if isinstance(key, str):
                key_text = key.strip().lower()
                if key_text.startswith("cuda:"):
                    index_text = key_text.split(":", maxsplit=1)[1]
                    normalized_key = int(index_text) if index_text.isdigit() else key_text
                elif key_text.isdigit():
                    normalized_key = int(key_text)
                elif key_text in {"cpu", "disk", "mps"}:
                    normalized_key = key_text
                else:
                    normalized_key = key
            resolved[normalized_key] = str(value)

        return resolved

    def _infer_model_input_device(self, model: nn.Module) -> torch.device:
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            try:
                embeddings = get_input_embeddings()
            except Exception:
                embeddings = None
            if embeddings is not None:
                weight = getattr(embeddings, "weight", None)
                if isinstance(weight, torch.Tensor):
                    return weight.device

        for parameter in model.parameters():
            return parameter.device
        return self.device

    def _load_transformer_backbone(self, strict: bool) -> nn.Module:
        base_model_id = self.model_cfg.get("base_model")
        if not base_model_id:
            if strict:
                raise ValueError("model.base_model must be set when use_mock_base_for_m2 is false")
            return self._build_mock_backbone()

        try:
            from transformers import AutoModelForCausalLM

            load_kwargs: dict[str, Any] = {
                "local_files_only": bool(self.model_cfg.get("local_files_only", True)),
            }
            load_kwargs.update(hf_extra_kwargs_from_model_cfg(self.model_cfg))

            resolved_dtype = self._resolve_torch_dtype()
            if resolved_dtype is not None:
                load_kwargs["dtype"] = resolved_dtype

            resolved_device_map = self._resolve_device_map()
            if resolved_device_map is not None:
                load_kwargs["device_map"] = resolved_device_map
                resolved_max_memory = self._resolve_max_memory()
                if resolved_max_memory is not None:
                    load_kwargs["max_memory"] = resolved_max_memory
                load_kwargs["low_cpu_mem_usage"] = bool(
                    self.model_cfg.get("low_cpu_mem_usage", True)
                )

            try:
                model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
            except TypeError as type_error:
                if resolved_dtype is not None and "dtype" in load_kwargs and "dtype" in str(type_error):
                    load_kwargs.pop("dtype", None)
                    load_kwargs["torch_dtype"] = resolved_dtype
                    model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
                else:
                    raise
            if resolved_device_map is None:
                model = model.to(self.device)
                self._is_model_sharded = False
            else:
                self._is_model_sharded = True

            self._base_model_input_device = self._infer_model_input_device(model)
            self.is_mock_base = False
            return model
        except Exception as exc:  # pragma: no cover - runtime/local-cache dependent.
            if strict:
                raise RuntimeError(f"Failed to load base model {base_model_id!r}") from exc
            return self._build_mock_backbone()

    def _freeze_base_model(self) -> None:
        if self.base_model is None:
            raise RuntimeError("base_model is not loaded yet")
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.frozen = True

    def _index_target_modules(self) -> None:
        if self.base_model is None:
            raise RuntimeError("base_model is not loaded yet")

        indexed: dict[str, list[nn.Linear]] = {layer_name: [] for layer_name in self.target_modules}
        group_by_module: dict[str, str] = {}
        module_specs: dict[str, LayerSpec] = {}
        runtime_candidates: list[tuple[str, nn.Linear, str, int]] = []
        for module_name, module in self.base_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            matched_segmented_group = next(
                (
                    layer_name
                    for layer_name in self.target_modules
                    if module_name == layer_name or module_name.endswith(f".{layer_name}")
                ),
                None,
            )
            if matched_segmented_group is not None:
                indexed[matched_segmented_group].append(module)
                indexed[module_name] = [module]
                group_by_module[module_name] = matched_segmented_group
                module_specs[module_name] = LayerSpec(
                    in_features=int(module.in_features),
                    out_features=int(module.out_features),
                )
                continue

            family = self._target_family_from_module_name(module_name)
            if family is None:
                continue

            layer_idx = self._extract_layer_index(module_name)
            if layer_idx is None:
                raise ValueError(
                    "Segmented target module contract requires runtime module names to include "
                    f"'layers.<idx>'. Got module_name={module_name!r}"
                )
            runtime_candidates.append((module_name, module, family, layer_idx))

        segment_map = self._build_segment_map([layer_idx for _, _, _, layer_idx in runtime_candidates])
        for module_name, module, family, layer_idx in runtime_candidates:
            segment = segment_map[layer_idx]
            group_key = self._build_segmented_group_key(segment=segment, family=family)
            indexed[group_key].append(module)
            indexed[module_name] = [module]
            group_by_module[module_name] = group_key
            module_specs[module_name] = LayerSpec(
                in_features=int(module.in_features),
                out_features=int(module.out_features),
            )

        self._target_linear_modules = indexed
        self._target_group_by_module = group_by_module
        self._module_specs = module_specs

        # Prefer runtime-derived dimensions for grouped target modules.
        for layer_name in self.target_modules:
            modules = indexed.get(layer_name, [])
            if not modules:
                continue
            linear = modules[0]
            self.layer_specs[layer_name] = LayerSpec(
                in_features=int(linear.in_features),
                out_features=int(linear.out_features),
            )

    def _generate_random_base_lora(self, rank: int) -> dict[str, dict[str, torch.Tensor]]:
        factors: dict[str, dict[str, torch.Tensor]] = {}
        for layer_name, spec in self.layer_specs.items():
            a0 = torch.randn(spec.out_features, rank, device=self.device) * 0.02
            b0 = torch.randn(rank, spec.in_features, device=self.device) * 0.02
            factors[layer_name] = {"A0": a0.detach(), "B0": b0.detach()}
        return factors

    def _load_ckpt_base_lora(self, ckpt_path: Path) -> dict[str, dict[str, torch.Tensor]]:
        payload = torch.load(ckpt_path, map_location=self.device)
        factors: dict[str, dict[str, torch.Tensor]] = {}

        if isinstance(payload, Mapping) and "layers" in payload and isinstance(payload["layers"], Mapping):
            for layer_name, layer_factors in payload["layers"].items():
                if layer_name not in self.target_modules or not isinstance(layer_factors, Mapping):
                    continue
                a0 = layer_factors.get("A0")
                b0 = layer_factors.get("B0")
                if isinstance(a0, torch.Tensor) and isinstance(b0, torch.Tensor):
                    factors[layer_name] = {"A0": a0.to(self.device), "B0": b0.to(self.device)}

        if not factors and isinstance(payload, Mapping):
            for layer_name in self.target_modules:
                a0_key = f"{layer_name}.A0"
                b0_key = f"{layer_name}.B0"
                if a0_key in payload and b0_key in payload:
                    a0 = payload[a0_key]
                    b0 = payload[b0_key]
                    if isinstance(a0, torch.Tensor) and isinstance(b0, torch.Tensor):
                        factors[layer_name] = {"A0": a0.to(self.device), "B0": b0.to(self.device)}

        return factors

    def _load_peft_base_lora(self, lora_path: str, strict: bool) -> dict[str, dict[str, torch.Tensor]]:
        if self.base_model is None:
            raise RuntimeError("base_model must be loaded before PEFT adapter")

        try:
            safetensors_path = Path(lora_path) / "adapter_model.safetensors"
            bin_path = Path(lora_path) / "adapter_model.bin"
            if safetensors_path.exists():
                from safetensors.torch import load_file as st_load

                state = st_load(str(safetensors_path), device=str(self.device))
            elif bin_path.exists():
                state = torch.load(str(bin_path), map_location=self.device)
            else:
                raise FileNotFoundError(
                    f"Neither adapter_model.safetensors nor adapter_model.bin found in {lora_path!r}"
                )
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to load PEFT adapter from {lora_path!r}") from exc
            return {}

        def _strip_known_prefix(module_name: str) -> str:
            normalized = module_name
            for prefix in (
                "base_model.model.model.",
                "base_model.model.",
                "model.model.",
                "model.",
            ):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            return normalized

        def _semantic_module_key(module_name: str) -> str:
            normalized = _strip_known_prefix(module_name)
            tokens = [token for token in normalized.split(".") if token and token != "model"]
            return ".".join(tokens)

        def _canonical_module_key(module_name: str) -> str:
            normalized = _strip_known_prefix(module_name)
            match = re.search(r"^(?P<prefix>.*?)(?P<tail>layers\.\d+\..+)$", normalized)
            if match is not None:
                prefix = match.group("prefix").rstrip(".")
                tail = match.group("tail")
                if not prefix:
                    return tail
                prefix_tokens = [token for token in prefix.split(".") if token and token != "model"]
                if not prefix_tokens:
                    return tail
                return ".".join(prefix_tokens + [tail])
            semantic_tokens = [token for token in normalized.split(".") if token and token != "model"]
            if semantic_tokens:
                return ".".join(semantic_tokens)
            return normalized

        def _tail_module_key(module_name: str) -> str:
            normalized = _strip_known_prefix(module_name)
            match = re.search(r"(layers\.\d+\..+)$", normalized)
            if match is not None:
                return match.group(1)
            semantic_tokens = [token for token in normalized.split(".") if token and token != "model"]
            if semantic_tokens:
                return ".".join(semantic_tokens[-2:]) if len(semantic_tokens) >= 2 else semantic_tokens[0]
            return normalized

        runtime_index: list[dict[str, Any]] = []
        runtime_exact_map: dict[str, list[dict[str, Any]]] = {}
        runtime_semantic_map: dict[str, list[dict[str, Any]]] = {}
        runtime_canonical_map: dict[str, list[dict[str, Any]]] = {}
        runtime_tail_map: dict[str, list[dict[str, Any]]] = {}
        for module_name in self._target_group_by_module:
            module_spec = self._module_specs.get(module_name)
            runtime_record = {
                "runtime_name": module_name,
                "normalized_name": _strip_known_prefix(module_name),
                "semantic_key": _semantic_module_key(module_name),
                "canonical_key": _canonical_module_key(module_name),
                "tail_key": _tail_module_key(module_name),
                "spec": module_spec,
            }
            runtime_index.append(runtime_record)
            runtime_exact_map.setdefault(runtime_record["normalized_name"], []).append(runtime_record)
            runtime_semantic_map.setdefault(runtime_record["semantic_key"], []).append(runtime_record)
            runtime_canonical_map.setdefault(runtime_record["canonical_key"], []).append(runtime_record)
            runtime_tail_map.setdefault(runtime_record["tail_key"], []).append(runtime_record)

        def _format_candidate(entry: dict[str, Any]) -> str:
            spec = entry.get("spec")
            if isinstance(spec, LayerSpec):
                return (
                    f"{entry['runtime_name']}(out={spec.out_features},in={spec.in_features})"
                )
            return str(entry["runtime_name"])

        def _resolve_runtime_candidates(peft_module_key: str) -> tuple[str, list[dict[str, Any]]]:
            normalized = _strip_known_prefix(peft_module_key)
            semantic = _semantic_module_key(peft_module_key)
            canonical = _canonical_module_key(peft_module_key)
            tail = _tail_module_key(peft_module_key)

            exact_candidates = runtime_exact_map.get(normalized, [])
            if exact_candidates:
                return "exact", exact_candidates

            semantic_candidates = runtime_semantic_map.get(semantic, [])
            if semantic_candidates:
                return "semantic", semantic_candidates

            canonical_candidates = runtime_canonical_map.get(canonical, [])
            if canonical_candidates:
                return "tower_aware_structural", canonical_candidates

            tail_candidates = runtime_tail_map.get(tail, [])
            if tail_candidates:
                return "suffix_structural", tail_candidates

            return "none", []

        def _is_configured_target_module(module_name: str) -> bool:
            normalized = _strip_known_prefix(module_name)
            semantic = _semantic_module_key(module_name)
            canonical = _canonical_module_key(module_name)
            if normalized in self.target_modules:
                return True
            return any(
                self._target_family_from_module_name(candidate) in _TARGET_FAMILIES
                for candidate in (normalized, semantic, canonical)
            )

        per_module_factors: dict[str, dict[str, torch.Tensor]] = {}
        per_module_sources: dict[str, str] = {}
        mapping_skips: list[str] = []
        grouped_factors: dict[str, dict[str, torch.Tensor]] = {}

        for key in state:
            if ".lora_A." not in key or not key.endswith(".weight"):
                continue
            module_key = key.split(".lora_A.", maxsplit=1)[0]
            lora_b_key = key.replace(".lora_A.", ".lora_B.")
            if lora_b_key not in state:
                continue
            if not _is_configured_target_module(module_key):
                continue

            match_stage, runtime_candidates = _resolve_runtime_candidates(module_key)
            if len(runtime_candidates) != 1:
                alignment_mode = self.peft_key_alignment
                strict_mode = strict
                if runtime_candidates:
                    candidate_text = [_format_candidate(item) for item in runtime_candidates]
                    message = (
                        f"Ambiguous PEFT-to-runtime mapping for {module_key!r}: "
                        f"stage={match_stage}, candidates={candidate_text}, "
                        f"alignment_mode={alignment_mode}, strict_mode={strict_mode}"
                    )
                    if strict:
                        raise ValueError(message)
                    mapping_skips.append(message)
                else:
                    message = (
                        f"Unmatched PEFT module key {module_key!r}: "
                        f"alignment_mode={alignment_mode}, strict_mode={strict_mode}"
                    )
                    if strict:
                        known_sample = [entry["runtime_name"] for entry in runtime_index[:3]]
                        raise ValueError(
                            f"{message}, known_runtime_samples={known_sample}"
                        )
                    mapping_skips.append(message)
                continue

            runtime_record = runtime_candidates[0]
            runtime_name = str(runtime_record["runtime_name"])
            previous_source = per_module_sources.get(runtime_name)
            if previous_source is not None and previous_source != module_key:
                message = (
                    f"Multiple PEFT keys map to the same runtime module {runtime_name!r}: "
                    f"first={previous_source!r}, second={module_key!r}, "
                    f"alignment_mode={self.peft_key_alignment}, strict_mode={strict}"
                )
                if strict:
                    raise ValueError(message)
                mapping_skips.append(message)
                continue
            lora_a_weight = state[key].to(self.device)
            lora_b_weight = state[lora_b_key].to(self.device)
            per_module_factors[runtime_name] = {
                "A0": lora_b_weight.detach(),
                "B0": lora_a_weight.detach(),
            }
            per_module_sources[runtime_name] = module_key
            group_key = self._target_group_by_module.get(runtime_name)
            if group_key is not None and group_key not in grouped_factors:
                grouped_factors[group_key] = {
                    "A0": lora_b_weight.detach(),
                    "B0": lora_a_weight.detach(),
                }

        if mapping_skips:
            warning_message = (
                "Skipped PEFT per-module mappings: "
                f"skipped_mapping_count={len(mapping_skips)}, "
                f"alignment_mode={self.peft_key_alignment}, strict_mode={strict}, "
                f"examples={mapping_skips[:3]}"
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

        if self.peft_key_alignment == "group_first":
            if strict and not grouped_factors:
                raise ValueError(
                    "No PEFT grouped factors matched runtime modules. "
                    "Check segmented target-module configuration and module naming compatibility."
                )
            return grouped_factors

        if strict and not per_module_factors:
            raise ValueError(
                "No PEFT per-module factors matched runtime modules. "
                "Check model.peft_key_alignment and module naming compatibility."
            )
        return per_module_factors if per_module_factors else grouped_factors

    def _load_base_lora_factors(self, lora_path: str | None, lora_source: str, strict: bool) -> None:
        factors: dict[str, dict[str, torch.Tensor]] = {}
        rank = int(self.model_cfg.get("base_lora_rank", 8))
        use_per_module_keys = lora_source == "peft" and self.peft_key_alignment != "group_first"
        expected_keys = list(self.target_modules)
        explicit_lora_path = bool(lora_path)

        if lora_path:
            if lora_source == "ckpt":
                candidate = Path(lora_path)
                if candidate.exists():
                    factors = self._load_ckpt_base_lora(candidate)
                elif strict:
                    raise FileNotFoundError(f"LoRA checkpoint not found: {candidate}")
            elif lora_source == "peft":
                factors = self._load_peft_base_lora(lora_path=lora_path, strict=strict)
            else:
                raise ValueError(f"Unsupported model.base_lora_source: {lora_source!r}")

        if not factors:
            if strict and explicit_lora_path and lora_source == "ckpt":
                raise ValueError(
                    "Checkpoint base LoRA must use the segmented six-key contract "
                    f"{list(_EXPECTED_SEGMENTED_TARGET_MODULES)}; "
                    f"no valid factors were found in checkpoint={lora_path!r}"
                )
            factors = self._generate_random_base_lora(rank=rank)
        elif use_per_module_keys:
            expected_keys = list(factors.keys())

        normalized: dict[str, dict[str, torch.Tensor]] = {}
        for layer_name in expected_keys:
            layer_factors = factors.get(layer_name)
            if layer_factors is None:
                spec = self._module_specs.get(layer_name, self.layer_specs.get(layer_name))
                if spec is None:
                    raise KeyError(f"Missing LayerSpec for {layer_name!r}")
                a0 = torch.randn(spec.out_features, rank, device=self.device) * 0.02
                b0 = torch.randn(rank, spec.in_features, device=self.device) * 0.02
                normalized[layer_name] = {"A0": a0.detach(), "B0": b0.detach()}
                continue

            a0 = layer_factors["A0"].to(self.device).detach()
            b0 = layer_factors["B0"].to(self.device).detach()
            if a0.dim() != 2 or b0.dim() != 2:
                raise ValueError(
                    f"{layer_name}: expected 2D A0/B0 tensors, got "
                    f"A0={tuple(a0.shape)}, B0={tuple(b0.shape)}"
                )
            if a0.shape[1] != b0.shape[0]:
                raise ValueError(
                    f"{layer_name}: invalid LoRA factors from source, "
                    f"A0={tuple(a0.shape)}, B0={tuple(b0.shape)}"
                )
            normalized[layer_name] = {"A0": a0, "B0": b0}

        self._base_lora_factors = normalized

        # Correct layer_specs from actual lora factor shapes: _index_target_modules may have
        # picked a wrong module first (e.g. vision tower in multimodal models), while the loaded
        # lora factors always reflect the intended text-tower dimensions.
        for layer_name, layer_facs in self._base_lora_factors.items():
            a0 = layer_facs["A0"]
            b0 = layer_facs["B0"]
            if layer_name in self.layer_specs:
                self.layer_specs[layer_name] = LayerSpec(
                    in_features=int(b0.shape[1]),
                    out_features=int(a0.shape[0]),
                )
            if layer_name in self._module_specs:
                self._module_specs[layer_name] = LayerSpec(
                    in_features=int(b0.shape[1]),
                    out_features=int(a0.shape[0]),
                )

        for group_name in self.target_modules:
            group_specs = {
                (int(layer_facs["B0"].shape[1]), int(layer_facs["A0"].shape[0]))
                for layer_name, layer_facs in self._base_lora_factors.items()
                if self._target_group_by_module.get(layer_name) == group_name
            }
            if not group_specs:
                continue
            if len(group_specs) != 1:
                raise ValueError(
                    f"{group_name}: inconsistent per-module LoRA shapes for grouped residual path: "
                    f"{sorted(group_specs)}"
                )
            in_features, out_features = next(iter(group_specs))
            self.layer_specs[group_name] = LayerSpec(
                in_features=in_features,
                out_features=out_features,
            )

        # Re-filter _target_linear_modules to only keep modules whose dimensions match the
        # now-correct layer_specs (filters out vision tower / other mismatched projections).
        for layer_name in list(self._target_linear_modules.keys()):
            spec = self._module_specs.get(layer_name, self.layer_specs.get(layer_name))
            if spec is None:
                continue
            self._target_linear_modules[layer_name] = [
                m
                for m in self._target_linear_modules[layer_name]
                if m.out_features == spec.out_features and m.in_features == spec.in_features
            ]


    def load_base_and_lora(
        self,
        base_model: nn.Module | None = None,
        lora_path: str | None = None,
        lora_source: str | None = None,
        strict: bool = False,
    ) -> nn.Module:
        if base_model is not None:
            if hasattr(base_model, "hf_device_map"):
                self.base_model = base_model
                self._is_model_sharded = True
            else:
                self.base_model = base_model.to(self.device)
                self._is_model_sharded = False
            self.is_mock_base = isinstance(self.base_model, MockBackbone)
            self._base_model_input_device = self._infer_model_input_device(self.base_model)
        else:
            if bool(self.model_cfg.get("use_mock_base_for_m2", True)):
                self.base_model = self._build_mock_backbone()
            else:
                self.base_model = self._load_transformer_backbone(strict=strict)

        self._index_target_modules()
        self._freeze_base_model()

        resolved_lora_source = str(lora_source or self.model_cfg.get("base_lora_source", "ckpt"))
        resolved_lora_path = lora_path or self.model_cfg.get("base_lora_path")
        self._load_base_lora_factors(
            lora_path=resolved_lora_path,
            lora_source=resolved_lora_source,
            strict=strict,
        )
        return self.base_model

    def base_model_input_device(self) -> torch.device:
        return self._base_model_input_device

    def uses_model_sharding(self) -> bool:
        return self._is_model_sharded

    def get_base_lora_factors(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            layer_name: {"A0": factors["A0"].clone(), "B0": factors["B0"].clone()}
            for layer_name, factors in self._base_lora_factors.items()
        }

    def derive_residual_a1_init_std_by_layer(self, tau: float) -> dict[str, float]:
        if not self._base_lora_factors:
            raise RuntimeError("base LoRA factors are not loaded")

        tau_value = float(tau)
        if tau_value < 0.0:
            raise ValueError(f"tau must be non-negative, got {tau!r}")

        grouped_a0_rms: dict[str, list[float]] = {layer_name: [] for layer_name in self.target_modules}
        for layer_name, factors in self._base_lora_factors.items():
            group_key = self._target_group_by_module.get(layer_name, layer_name)
            if group_key not in grouped_a0_rms:
                continue
            a0 = factors["A0"]
            rms = torch.sqrt(torch.mean(torch.square(a0.to(dtype=torch.float32)))).item()
            grouped_a0_rms[group_key].append(rms)

        missing = [layer_name for layer_name, values in grouped_a0_rms.items() if not values]
        if missing:
            raise RuntimeError(
                "Cannot derive residual A1 init stds for all target layers; "
                f"missing_group_keys={missing}"
            )

        return {
            layer_name: tau_value * float(sum(values) / len(values))
            for layer_name, values in grouped_a0_rms.items()
        }

    def apply_residual(
        self,
        residual_factors: Mapping[str, Any],
        alpha: Any,
    ) -> dict[str, torch.Tensor]:
        if not self._base_lora_factors:
            raise RuntimeError("base LoRA factors are not loaded")
        if "lora_A1" not in residual_factors or "lora_B1" not in residual_factors:
            raise ValueError("residual_factors must include 'lora_A1' and 'lora_B1'")

        a0 = {layer: factors["A0"] for layer, factors in self._base_lora_factors.items()}
        b0 = {layer: factors["B0"] for layer, factors in self._base_lora_factors.items()}
        raw_a1 = residual_factors["lora_A1"]
        raw_b1 = residual_factors["lora_B1"]

        if not isinstance(raw_a1, Mapping) or not isinstance(raw_b1, Mapping):
            raise TypeError("residual_factors lora_A1/lora_B1 must be Mapping[str, torch.Tensor]")

        base_keys = set(a0.keys())
        source_a1 = dict(raw_a1)
        source_b1 = dict(raw_b1)
        supported_group_keys = set(self.target_modules)

        unsupported_residual_keys = sorted(
            {
                key
                for key in set(source_a1) | set(source_b1)
                if key not in base_keys and key not in supported_group_keys
            }
        )
        if unsupported_residual_keys:
            raise ValueError(
                "residual_factors keys must use per-module runtime keys or the segmented six-key "
                f"contract {list(_EXPECTED_SEGMENTED_TARGET_MODULES)}; "
                f"unsupported_keys={unsupported_residual_keys[:3]}"
            )

        if set(source_a1.keys()) == base_keys and set(source_b1.keys()) == base_keys:
            expanded_a1 = source_a1
            expanded_b1 = source_b1
        else:
            expanded_a1: dict[str, torch.Tensor] = {}
            expanded_b1: dict[str, torch.Tensor] = {}
            missing_for_base: list[str] = []
            for base_key in a0:
                if base_key in source_a1 and base_key in source_b1:
                    expanded_a1[base_key] = source_a1[base_key]
                    expanded_b1[base_key] = source_b1[base_key]
                    continue

                group_key = self._target_group_by_module.get(base_key)
                if (
                    group_key is not None
                    and group_key in source_a1
                    and group_key in source_b1
                ):
                    expanded_a1[base_key] = source_a1[group_key]
                    expanded_b1[base_key] = source_b1[group_key]
                    continue
                missing_for_base.append(base_key)

            if missing_for_base:
                raise ValueError(
                    "residual_factors missing keys for base LoRA layers. "
                    f"missing examples={missing_for_base[:3]}"
                )

        if isinstance(alpha, Mapping):
            unsupported_alpha_keys = sorted(
                {
                    key
                    for key in alpha
                    if key not in base_keys and key not in supported_group_keys
                }
            )
            if unsupported_alpha_keys:
                raise ValueError(
                    "alpha keys must use per-module runtime keys or the segmented six-key contract "
                    f"{list(_EXPECTED_SEGMENTED_TARGET_MODULES)}; "
                    f"unsupported_keys={unsupported_alpha_keys[:3]}"
                )
            expanded_alpha: dict[str, Any] = {}
            for base_key in a0:
                if base_key in alpha:
                    expanded_alpha[base_key] = alpha[base_key]
                    continue
                group_key = self._target_group_by_module.get(base_key)
                if group_key is not None and group_key in alpha:
                    expanded_alpha[base_key] = alpha[group_key]
                    continue
                raise ValueError(
                    f"alpha missing key for base LoRA layer {base_key!r} "
                    f"(group={group_key!r})"
                )
        else:
            expanded_alpha = {base_key: alpha for base_key in a0}

        delta_weight = compose_delta(
            a0=a0,
            b0=b0,
            a1=expanded_a1,
            b1=expanded_b1,
            alpha=expanded_alpha,
        )
        self._latest_delta_weight = delta_weight
        return delta_weight

    def clear_injection(self) -> None:
        for handle in self._injection_handles:
            handle.remove()
        self._injection_handles.clear()

    def inject_delta(
        self,
        delta_weight: Mapping[str, torch.Tensor] | None = None,
        mode: str = "forward_hook",
    ) -> int:
        if mode != "forward_hook":
            raise ValueError(f"Unsupported inject mode: {mode!r}")
        if self.base_model is None:
            raise RuntimeError("base_model is not loaded")

        self.clear_injection()
        payload = dict(delta_weight or self._latest_delta_weight)
        if not payload:
            raise ValueError("delta_weight is empty; call apply_residual first")

        def _build_hook(layer_delta: torch.Tensor):
            def _hook(_: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
                if not args:
                    raise RuntimeError("Expected linear input tensor in hook args")
                inputs = args[0]
                if not isinstance(inputs, torch.Tensor):
                    raise RuntimeError("Linear hook input must be torch.Tensor")

                delta = layer_delta.to(device=inputs.device, dtype=inputs.dtype)
                if delta.dim() == 2:
                    return output + F.linear(inputs, delta)

                if delta.dim() != 3:
                    raise ValueError(
                        f"Injected delta must be 2D/3D tensor, got shape={tuple(delta.shape)}"
                    )
                if inputs.shape[0] != delta.shape[0]:
                    raise ValueError(
                        f"Batch mismatch: input batch={inputs.shape[0]} delta batch={delta.shape[0]}"
                    )

                if inputs.dim() == 2:
                    addition = torch.bmm(inputs.unsqueeze(1), delta.transpose(1, 2)).squeeze(1)
                    return output + addition
                if inputs.dim() == 3:
                    addition = torch.einsum("bti,boi->bto", inputs, delta)
                    return output + addition
                raise ValueError(
                    f"Unsupported input rank for injected delta: {inputs.dim()} "
                    "(expected 2D or 3D tensor)"
                )

            return _hook

        handle_count = 0
        for layer_name, layer_delta in payload.items():
            target_modules = self._target_linear_modules.get(layer_name, [])
            if not target_modules:
                raise ValueError(f"No target module resolved for layer {layer_name!r}")
            for module in target_modules:
                handle = module.register_forward_hook(_build_hook(layer_delta))
                self._injection_handles.append(handle)
                handle_count += 1
        return handle_count
