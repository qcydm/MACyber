from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from collections.abc import Mapping

import torch

from dhr.data.convert_real import load_real_dataset
from dhr.data.dataset import AttackDataset
from dhr.data.synthetic import generate_synthetic_dataset
from dhr.modeling.base_adapter import BaseAdapterManager
from dhr.modeling.hyper_residual import HyperLayerShape, HyperResidualNet
from dhr.modeling.merge_hypernet import MergeHyperNet
from dhr.modeling.lora_ops import compute_lora_weight
from dhr.modeling.prompt_feature_projection import (
    load_prompt_feature_projection,
    resolve_prompt_feature_hidden_size,
)
from dhr.modeling.retriever import FingerprintRetriever
from dhr.utils.hf_kwargs import hf_extra_kwargs_from_model_cfg
from dhr.utils.model_forward import last_hidden_state_from_base_model
from dhr.utils.prompt_features import build_prompt_text, pack_hidden_features


def _experiment_name(config: dict[str, Any]) -> str:
    project_cfg = config.get("project", {})
    return str(config.get("experiment", {}).get("name", project_cfg.get("name", "default")))


def _resolve_checkpoint(config: dict[str, Any], checkpoint: str | None) -> Path | None:
    if checkpoint:
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    project_cfg = config.get("project", {})
    ckpt_root = Path(str(project_cfg.get("checkpoint_root", "checkpoints")))
    default_path = (ckpt_root / _experiment_name(config) / "best.pt").resolve()
    return default_path if default_path.exists() else None


def _load_checkpoint_if_available(
    checkpoint_path: Path | None,
    hyper_residual: HyperResidualNet,
    merge_hypernet: MergeHyperNet,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None

    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    if "hyper_residual" not in state or "merge_hypernet" not in state:
        raise ValueError(
            f"Checkpoint missing keys 'hyper_residual'/'merge_hypernet': {checkpoint_path}"
        )

    hyper_residual.load_state_dict(state["hyper_residual"], strict=True)
    merge_hypernet.load_state_dict(state["merge_hypernet"], strict=True)
    return state


def _load_input_payload(input_path: str | None) -> dict[str, Any]:
    if not input_path:
        return {}
    sample_path = Path(input_path)
    if not sample_path.exists():
        return {}

    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"adapt input must be a JSON object: {sample_path}")
    return payload


def _extract_query_record(payload: dict[str, Any]) -> dict[str, str] | None:
    instruction = str(payload.get("instruction", "")).strip()
    inp = str(payload.get("input", "")).strip()
    if not instruction and not inp:
        return None
    return {
        "attack_family": str(payload.get("attack_family", "unknown")),
        "instruction": instruction,
        "input": inp,
    }


def _to_feature_tensor(
    sample_features_raw: Any,
    feature_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if sample_features_raw is None:
        return None
    sample_features = torch.tensor(sample_features_raw, dtype=torch.float32, device=device)
    if sample_features.dim() == 1:
        sample_features = sample_features.unsqueeze(0)
    if sample_features.dim() != 2:
        raise ValueError(
            f"sample_features must be 1D/2D tensor-like, got shape={tuple(sample_features.shape)}"
        )
    if sample_features.shape[1] != feature_dim:
        raise ValueError(
            f"sample_features dim mismatch: expected {feature_dim}, got {sample_features.shape[1]}"
        )
    return sample_features


def _to_memory_tokens(
    memory_tokens_raw: Any,
    feature_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if memory_tokens_raw is None:
        return None
    memory_tokens = torch.tensor(memory_tokens_raw, dtype=torch.float32, device=device)
    if memory_tokens.dim() == 1:
        memory_tokens = memory_tokens.unsqueeze(0)
    if memory_tokens.dim() == 3:
        if memory_tokens.shape[0] != 1:
            raise ValueError(
                "memory_tokens with 3D shape must have batch=1 in adapt_once flow"
            )
        memory_tokens = memory_tokens[0]
    if memory_tokens.dim() != 2:
        raise ValueError(
            f"memory_tokens must be 1D/2D/3D tensor-like, got shape={tuple(memory_tokens.shape)}"
        )
    if memory_tokens.shape[1] != feature_dim:
        raise ValueError(
            f"memory_tokens dim mismatch: expected {feature_dim}, got {memory_tokens.shape[1]}"
        )
    return memory_tokens


def _to_retriever_stats(
    stats_raw: Any,
    stats_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if stats_raw is None:
        return None
    stats = torch.tensor(stats_raw, dtype=torch.float32, device=device)
    if stats.dim() == 1:
        stats = stats.unsqueeze(0)
    if stats.dim() != 2:
        raise ValueError(
            f"retriever_stats must be 1D/2D tensor-like, got shape={tuple(stats.shape)}"
        )
    if stats.shape[1] != stats_dim:
        raise ValueError(
            f"retriever_stats dim mismatch: expected {stats_dim}, got {stats.shape[1]}"
        )
    return stats


def _build_tokenizer_if_needed(config: dict[str, Any], adapter_manager: BaseAdapterManager) -> Any:
    if adapter_manager.is_mock_base:
        return None

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - import environment dependent.
        raise RuntimeError("transformers is required for real backbone adaptation") from exc

    model_cfg = config.get("model", {})
    base_model_id = str(model_cfg.get("base_model", ""))
    if not base_model_id:
        raise ValueError("model.base_model must be set for real backbone adaptation")
    tok_kw: dict[str, Any] = {
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    tok_kw.update(hf_extra_kwargs_from_model_cfg(model_cfg))
    return AutoTokenizer.from_pretrained(base_model_id, **tok_kw)


def _forward_logits_for_sample(
    record: dict[str, str] | None,
    sample_features: torch.Tensor,
    adapter_manager: BaseAdapterManager,
    tokenizer: Any,
    max_seq_len: int,
) -> torch.Tensor:
    if adapter_manager.base_model is None:
        raise RuntimeError("base model is not initialized")

    if adapter_manager.is_mock_base:
        output = adapter_manager.base_model(sample_features)
        logits = output["logits"]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        return logits

    if record is None:
        return torch.zeros(
            sample_features.shape[0],
            1,
            device=adapter_manager.device,
            dtype=torch.float32,
        )
    if tokenizer is None:
        raise RuntimeError("tokenizer is required for real backbone adaptation")

    tokenized = tokenizer(
        [build_prompt_text(record)],
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_device = adapter_manager.base_model_input_device()
    tokenized = {key: value.to(input_device) for key, value in tokenized.items()}
    model_out = adapter_manager.base_model(**tokenized)
    if isinstance(model_out, Mapping):
        logits = model_out.get("logits")
    else:
        logits = getattr(model_out, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    hidden = last_hidden_state_from_base_model(adapter_manager.base_model, **tokenized)
    return hidden.mean(dim=1)


def _encode_prompt_derived_features(
    record: dict[str, str],
    *,
    adapter_manager: BaseAdapterManager,
    tokenizer: Any,
    prompt_feature_proj: torch.nn.Module | None,
    max_seq_len: int,
) -> torch.Tensor:
    if adapter_manager.is_mock_base:
        raise RuntimeError("prompt-derived features are not available for mock backbone")
    if tokenizer is None:
        raise RuntimeError("tokenizer is required for prompt-derived features")
    if prompt_feature_proj is None:
        raise RuntimeError("prompt_feature_proj is required for prompt-derived features")

    prompt_text = build_prompt_text(record)
    tokenized = tokenizer(
        [prompt_text],
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_device = adapter_manager.base_model_input_device()
    tokenized = {key: value.to(input_device) for key, value in tokenized.items()}
    with torch.no_grad():
        hidden = last_hidden_state_from_base_model(adapter_manager.base_model, **tokenized)
    packed = pack_hidden_features(hidden=hidden, attention_mask=tokenized.get("attention_mask"))
    packed_f32 = packed.to(dtype=torch.float32, device=adapter_manager.device)
    return prompt_feature_proj(packed_f32)


def _resolve_conditioning_inputs(
    *,
    input_payload: dict[str, Any],
    query_record: dict[str, str] | None,
    adapter_manager: BaseAdapterManager,
    retriever: FingerprintRetriever,
    tokenizer: Any,
    prompt_feature_proj: torch.nn.Module | None,
    feature_dim: int,
    stats_dim: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, Any], dict[str, str]]:
    device = adapter_manager.device
    sample_features = _to_feature_tensor(
        input_payload.get("sample_features"),
        feature_dim=feature_dim,
        device=device,
    )
    memory_tokens = _to_memory_tokens(
        input_payload.get("memory_tokens"),
        feature_dim=feature_dim,
        device=device,
    )
    retriever_stats = _to_retriever_stats(
        input_payload.get("retriever_stats"),
        stats_dim=stats_dim,
        device=device,
    )

    sources = {
        "sample_features": "payload" if sample_features is not None else "",
        "memory_tokens": "payload" if memory_tokens is not None else "",
        "retriever_stats": "payload" if retriever_stats is not None else "",
    }
    retrieval_meta: dict[str, Any] = {
        "used_retriever": False,
        "cache_hit": False,
        "fallback": False,
        "topk_size": 0,
    }
    retrieved: dict[str, Any] | None = None
    if query_record is not None:
        retrieved = retriever.retrieve(query_record)
        retrieval_meta = {
            "used_retriever": True,
            "cache_hit": bool(retrieved.get("cache_hit", False)),
            "fallback": bool(retrieved.get("fallback", False)),
            "topk_size": len(retrieved.get("topk_ids", [])),
        }

    if sample_features is None:
        if query_record is not None and not adapter_manager.is_mock_base:
            sample_features = _encode_prompt_derived_features(
                query_record,
                adapter_manager=adapter_manager,
                tokenizer=tokenizer,
                prompt_feature_proj=prompt_feature_proj,
                max_seq_len=max_seq_len,
            )
            sources["sample_features"] = "prompt_derived"
        elif retrieved is not None:
            sample_features = retrieved["query_embedding"].unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )
            sources["sample_features"] = "retriever_query_embedding"
        else:
            sample_features = torch.randn(1, feature_dim, device=device, dtype=torch.float32)
            sources["sample_features"] = "random_fallback"

    if memory_tokens is None and retrieved is not None:
        memory_raw = retrieved.get("memory_tokens")
        if isinstance(memory_raw, torch.Tensor):
            memory_tokens = memory_raw.to(device=device, dtype=torch.float32)
            sources["memory_tokens"] = "retriever"
    if not sources["memory_tokens"]:
        sources["memory_tokens"] = "none"

    if retriever_stats is None and retrieved is not None:
        stats_raw = retrieved.get("retriever_stats")
        if isinstance(stats_raw, torch.Tensor):
            retriever_stats = stats_raw.unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            )
            sources["retriever_stats"] = "retriever"
    if retriever_stats is None:
        retriever_stats = torch.zeros(
            1,
            stats_dim,
            device=device,
            dtype=sample_features.dtype,
        )
        sources["retriever_stats"] = "zeros"

    if retriever_stats.shape[0] == 1 and sample_features.shape[0] > 1:
        retriever_stats = retriever_stats.expand(sample_features.shape[0], -1)
    elif retriever_stats.shape[0] != sample_features.shape[0]:
        raise ValueError(
            "retriever_stats batch mismatch: "
            f"sample_features batch={sample_features.shape[0]} "
            f"vs retriever_stats batch={retriever_stats.shape[0]}"
        )

    return sample_features, memory_tokens, retriever_stats, retrieval_meta, sources


def _tensor_to_python_scalar(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return float(value.item())
        return float(value.mean().item())
    return float(value)


def run_adaptation(
    config: dict[str, Any],
    input_path: str | None = None,
    checkpoint: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    retriever_cfg = config.get("retriever", {})
    training_cfg = config.get("training", {})
    feature_dim = int(model_cfg.get("sample_feature_dim", 64))
    stats_dim = int(retriever_cfg.get("stats_dim", 2))
    max_seq_len = int(training_cfg.get("max_seq_len", 256))

    adapter_manager = BaseAdapterManager(config)
    adapter_manager.load_base_and_lora(strict=False)
    layer_shapes: dict[str, HyperLayerShape] = {
        layer_name: HyperLayerShape(
            in_features=spec.in_features,
            out_features=spec.out_features,
        )
        for layer_name, spec in adapter_manager.layer_specs.items()
    }
    residual_a1_init_tau = float(model_cfg.get("residual_a1_init_tau", 0.05))
    a_head_init_std_by_layer = adapter_manager.derive_residual_a1_init_std_by_layer(
        tau=residual_a1_init_tau
    )

    hyper_residual = HyperResidualNet(
        config,
        layer_shapes=layer_shapes,
        a_head_init_std_by_layer=a_head_init_std_by_layer,
    ).to(adapter_manager.device)
    merge_hypernet = MergeHyperNet(config, layer_names=list(layer_shapes.keys())).to(
        adapter_manager.device
    )
    checkpoint_path = _resolve_checkpoint(config=config, checkpoint=checkpoint)
    checkpoint_state = _load_checkpoint_if_available(
        checkpoint_path=checkpoint_path,
        hyper_residual=hyper_residual,
        merge_hypernet=merge_hypernet,
        device=adapter_manager.device,
    )
    checkpoint_loaded = checkpoint_state is not None
    hyper_residual.eval()
    merge_hypernet.eval()

    retriever = FingerprintRetriever.from_config(config)
    retriever.cache_enabled = bool(use_cache)
    if config.get("data", {}).get("real", {}).get("source_json"):
        real_cfg = config.get("data", {}).get("real", {})
        real_train_path = Path(str(real_cfg.get("output_dir", "data/real"))) / "train.jsonl"
        if real_train_path.exists():
            train_dataset = AttackDataset.from_jsonl(real_train_path)
        else:
            manifest = load_real_dataset(config)
            train_dataset = AttackDataset.from_manifest(manifest, split="train")
    else:
        synthetic_cfg = config.get("data", {}).get("synthetic", {})
        train_path = Path(str(synthetic_cfg.get("output_dir", "data/synthetic"))) / "train.jsonl"
        if train_path.exists():
            train_dataset = AttackDataset.from_jsonl(train_path)
        else:
            manifest = generate_synthetic_dataset(config)
            train_dataset = AttackDataset.from_manifest(manifest, split="train")
    retriever.build_index(train_dataset.records)

    input_payload = _load_input_payload(input_path=input_path)
    query_record = _extract_query_record(input_payload)
    tokenizer = _build_tokenizer_if_needed(config=config, adapter_manager=adapter_manager)
    prompt_feature_proj = None
    if (
        query_record is not None
        and input_payload.get("sample_features") is None
        and not adapter_manager.is_mock_base
    ):
        if checkpoint_state is None:
            raise ValueError(
                "Prompt-derived sample_features require a checkpoint with prompt_feature_proj metadata"
            )
        hidden_size = resolve_prompt_feature_hidden_size(
            adapter_manager.base_model,
            fallback_hidden_size=feature_dim,
        )
        prompt_feature_proj = load_prompt_feature_projection(
            checkpoint_state,
            hidden_size=hidden_size,
            feature_dim=feature_dim,
            device=adapter_manager.device,
            checkpoint_path=checkpoint_path,
            required=True,
        )

    sample_features, memory_tokens, retriever_stats, retrieval_meta, feature_sources = (
        _resolve_conditioning_inputs(
            input_payload=input_payload,
            query_record=query_record,
            adapter_manager=adapter_manager,
            retriever=retriever,
            tokenizer=tokenizer,
            prompt_feature_proj=prompt_feature_proj,
            feature_dim=feature_dim,
            stats_dim=stats_dim,
            max_seq_len=max_seq_len,
        )
    )

    start_time = time.perf_counter()
    with torch.no_grad():
        hyper_out = hyper_residual(sample_features=sample_features, memory_tokens=memory_tokens)
        alpha = merge_hypernet(hidden=hyper_out["hidden"], retriever_stats=retriever_stats)
        delta_weight = adapter_manager.apply_residual(residual_factors=hyper_out, alpha=alpha)
        injected_layer_count = adapter_manager.inject_delta(delta_weight=delta_weight, mode="forward_hook")
        try:
            logits = _forward_logits_for_sample(
                record=query_record,
                sample_features=sample_features,
                adapter_manager=adapter_manager,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            )
        finally:
            adapter_manager.clear_injection()

        score_tensor = torch.sigmoid(logits.view(-1).mean())

    effective_rank = {}
    for layer_name in hyper_out["lora_A1"]:
        residual_weight = compute_lora_weight(
            hyper_out["lora_A1"][layer_name],
            hyper_out["lora_B1"][layer_name],
        )
        if residual_weight.dim() == 3:
            residual_weight = residual_weight[0]
        effective_rank[layer_name] = int(torch.linalg.matrix_rank(residual_weight).item())

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    score = float(score_tensor.item())
    prediction = int(score >= 0.5)

    return {
        "status": "adapt_ok",
        "input_path": input_path,
        "checkpoint_used": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_loaded": checkpoint_loaded,
        "use_cache": use_cache,
        "retriever": retrieval_meta,
        "retriever_index_size": retriever.index_size,
        "sample_features_source": feature_sources["sample_features"],
        "memory_tokens_source": feature_sources["memory_tokens"],
        "retriever_stats_source": feature_sources["retriever_stats"],
        "sample_features_shape": list(sample_features.shape),
        "alpha": {layer_name: _tensor_to_python_scalar(v) for layer_name, v in alpha.items()},
        "effective_rank": effective_rank,
        "latency_ms": latency_ms,
        "prediction": prediction,
        "score": score,
        "generated_layers": list(delta_weight.keys()),
        "injected_layer_count": injected_layer_count,
    }
