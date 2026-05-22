from __future__ import annotations

from collections.abc import Mapping
import csv
import json
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn as nn
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback.
    tqdm = None

from dhr.data.dataset import AttackDataset
from dhr.data.convert_real import load_real_dataset
from dhr.data.synthetic import generate_synthetic_dataset
from dhr.data.schemas import parse_output_json
from dhr.eval.baselines import (
    BASELINE_RESIDUAL_NO_RETRIEVER,
    BASELINE_RESIDUAL_WITH_RETRIEVER,
    BASELINE_STATIC,
    resolve_baselines,
)
from dhr.eval.metrics import (
    compute_generative_metrics,
    compute_latency_stats,
    render_table,
)
from dhr.modeling.base_adapter import BaseAdapterManager
from dhr.modeling.hyper_residual import HyperLayerShape, HyperResidualNet
from dhr.modeling.lora_ops import compute_lora_weight
from dhr.modeling.merge_hypernet import MergeHyperNet
from dhr.modeling.prompt_feature_projection import (
    load_prompt_feature_projection,
    resolve_prompt_feature_hidden_size,
)
from dhr.modeling.retriever import FingerprintRetriever
from dhr.utils.hf_kwargs import hf_extra_kwargs_from_model_cfg
from dhr.utils.io import ensure_dir
from dhr.utils.model_forward import last_hidden_state_from_base_model
from dhr.utils.prompt_features import build_prompt_text, pack_hidden_features

TABLE_COLUMNS = [
    "baseline",
    "split",
    "sample_count",
    "action_accuracy",
    "severity_accuracy",
    "official_accuracy",
    "exact_match",
    "adapt_latency_avg_ms",
    "adapt_latency_p95_ms",
    "peak_vram_mb",
    "effective_rank_mean",
    "effective_rank_max",
    "ood_gap",
    "memory_mode",
]
EVAL_SPLITS = ("test_id", "test_ood")
_STRICT_JSON_ARRAY_BLOCK_RE = re.compile(r"```json\s*(\[.*?\])\s*```", re.DOTALL)
_ANY_JSON_FENCE_RE = re.compile(r"```json\b", re.IGNORECASE)


def _looks_like_bare_json_container(text: str) -> bool:
    if not text:
        return False
    return (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    )


def _classify_parse_status(generated_text: str, pred_parsed: Mapping[str, Any] | None) -> str:
    if pred_parsed is not None:
        return "ok"
    if _STRICT_JSON_ARRAY_BLOCK_RE.search(generated_text):
        return "json_decode_failed_or_empty"
    if _ANY_JSON_FENCE_RE.search(generated_text):
        return "json_fence_present_but_format_mismatch"
    if _looks_like_bare_json_container(generated_text.strip()):
        return "bare_json_decode_failed_or_empty"
    return "missing_json_block"


def _use_eval_tqdm(config: Mapping[str, Any]) -> bool:
    eval_cfg = config.get("eval", {})
    if not isinstance(eval_cfg, Mapping):
        return tqdm is not None
    return bool(eval_cfg.get("use_tqdm", True)) and tqdm is not None


def _eval_tiny_first_n(config: Mapping[str, Any]) -> int:
    eval_cfg = config.get("eval", {})
    if not isinstance(eval_cfg, Mapping):
        return 0

    raw = eval_cfg.get("tiny_first_n", 0)
    if raw in (None, ""):
        return 0

    try:
        tiny_first_n = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"eval.tiny_first_n must be an integer, got: {raw!r}") from exc

    if tiny_first_n < 0:
        raise ValueError(f"eval.tiny_first_n must be >= 0, got: {tiny_first_n}")
    return tiny_first_n


def _experiment_name(config: dict[str, Any]) -> str:
    project_cfg = config.get("project", {})
    return str(config.get("experiment", {}).get("name", project_cfg.get("name", "default")))


def _resolve_checkpoint(config: dict[str, Any], checkpoint: str | None) -> Path:
    def _pick_checkpoint_file(ckpt_dir: Path) -> Path:
        for candidate_name in ("best.pt", "last.pt"):
            candidate_path = ckpt_dir / candidate_name
            if candidate_path.is_file():
                return candidate_path.resolve()
        raise FileNotFoundError(
            f"checkpoint directory does not contain best.pt or last.pt: {ckpt_dir.resolve()}"
        )

    if checkpoint:
        ckpt_path = Path(checkpoint).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        if ckpt_path.is_dir():
            return _pick_checkpoint_file(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint is not a file: {ckpt_path}")
        return ckpt_path

    project_cfg = config.get("project", {})
    ckpt_root = Path(str(project_cfg.get("checkpoint_root", "checkpoints")))
    ckpt_dir = (ckpt_root / _experiment_name(config)).resolve()
    try:
        return _pick_checkpoint_file(ckpt_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"checkpoint not found and fallback failed under: {ckpt_dir}. "
            "Provide --checkpoint <file_or_dir> or ensure best.pt/last.pt exists."
        ) from exc


def _build_tokenizer(config: dict[str, Any], adapter_manager: BaseAdapterManager) -> Any:
    if adapter_manager.is_mock_base:
        return None

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for real backbone evaluation") from exc

    model_cfg = config.get("model", {})
    base_model_id = str(model_cfg.get("base_model", ""))
    if not base_model_id:
        raise ValueError("model.base_model must be set for real backbone evaluation")

    tok_kw: dict[str, Any] = {
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    tok_kw.update(hf_extra_kwargs_from_model_cfg(model_cfg))
    return AutoTokenizer.from_pretrained(base_model_id, **tok_kw)


def _build_prompt_feature_proj(
    config: dict[str, Any],
    adapter_manager: BaseAdapterManager,
    checkpoint_state: dict[str, Any],
    checkpoint_path: Path,
    *,
    required: bool,
) -> nn.Linear | None:
    """Reconstruct prompt_feature_proj from checkpoint if present."""
    if adapter_manager.is_mock_base:
        return None

    feature_dim = int(config.get("model", {}).get("sample_feature_dim", 64))
    hidden_size = resolve_prompt_feature_hidden_size(
        adapter_manager.base_model,
        fallback_hidden_size=feature_dim,
    )
    return load_prompt_feature_projection(
        checkpoint_state,
        hidden_size=hidden_size,
        feature_dim=feature_dim,
        device=adapter_manager.device,
        checkpoint_path=checkpoint_path,
        required=required,
    )


def _encode_prompt_features(
    prompt_texts: list[str],
    adapter_manager: BaseAdapterManager,
    tokenizer: Any,
    retriever: FingerprintRetriever,
    prompt_feature_proj: nn.Linear | None,
    max_seq_len: int,
) -> torch.Tensor:
    """Compute sample_features for the evaluator (no-grad LLM forward or hash fallback)."""
    if adapter_manager.is_mock_base or tokenizer is None:
        vecs = [
            retriever.encode_query(text).to(dtype=torch.float32)
            for text in prompt_texts
        ]
        return torch.stack(vecs, dim=0).to(adapter_manager.device)
    if prompt_feature_proj is None:
        raise RuntimeError("prompt_feature_proj is required for real-backbone prompt features")

    input_device = adapter_manager.base_model_input_device()
    tokenized = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    tokenized = {k: v.to(input_device) for k, v in tokenized.items()}

    with torch.no_grad():
        hidden = last_hidden_state_from_base_model(adapter_manager.base_model, **tokenized)

    packed = pack_hidden_features(hidden=hidden, attention_mask=tokenized.get("attention_mask"))
    packed_f32 = packed.to(dtype=torch.float32, device=adapter_manager.device)
    return prompt_feature_proj(packed_f32)  # [B, feature_dim]


def _generate_response(
    record: dict[str, Any],
    adapter_manager: BaseAdapterManager,
    tokenizer: Any,
    max_seq_len: int,
    max_new_tokens: int,
) -> str:
    """Generate model response for a single record with injected LoRA delta."""
    if adapter_manager.is_mock_base or tokenizer is None:
        return ""

    instruction = str(record.get("instruction", ""))
    inp = str(record.get("input", ""))

    has_chat_template = hasattr(tokenizer, "apply_chat_template") and (
        tokenizer.chat_template is not None
    )

    if has_chat_template:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inp},
        ]
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(templated, torch.Tensor):
            prompt_ids = templated
            attention_mask = torch.ones_like(prompt_ids)
        elif isinstance(templated, Mapping):
            if "input_ids" not in templated:
                raise ValueError("apply_chat_template output missing 'input_ids'")
            prompt_ids = templated["input_ids"]
            attention_mask = templated.get("attention_mask")
        else:
            raise TypeError(
                f"Unsupported apply_chat_template return type: {type(templated).__name__}"
            )
    else:
        prompt_text = f"{instruction}\n\n{inp}\n\n" if inp else f"{instruction}\n\n"
        tokenized = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        prompt_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask")

    # Truncate prompt if too long
    if prompt_ids.shape[1] > max_seq_len:
        prompt_ids = prompt_ids[:, -max_seq_len:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -max_seq_len:]

    input_device = adapter_manager.base_model_input_device()
    prompt_ids = prompt_ids.to(input_device)
    if attention_mask is None:
        attention_mask = torch.ones_like(prompt_ids)
    else:
        attention_mask = attention_mask.to(input_device)

    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    with torch.no_grad():
        generated = adapter_manager.base_model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _sample_effective_rank(hyper_out: dict[str, Any]) -> float:
    ranks: list[float] = []
    for layer_name, a_factor in hyper_out["lora_A1"].items():
        factor = a_factor.float()
        if factor.dim() == 2:
            rank_value = torch.linalg.matrix_rank(factor).item()
        else:
            rank_value = max(
                torch.linalg.matrix_rank(factor[idx]).item()
                for idx in range(factor.shape[0])
            )
        ranks.append(float(rank_value))
    return sum(ranks) / float(max(len(ranks), 1))


def _static_delta_weight(adapter_manager: BaseAdapterManager) -> dict[str, torch.Tensor]:
    base_factors = adapter_manager.get_base_lora_factors()
    return {
        layer_name: compute_lora_weight(factors["A0"], factors["B0"])
        for layer_name, factors in base_factors.items()
    }


def _evaluate_split(
    config: dict[str, Any],
    baseline: str,
    split: str,
    records: list[dict[str, Any]],
    adapter_manager: BaseAdapterManager,
    hyper_residual: HyperResidualNet,
    merge_hypernet: MergeHyperNet,
    retriever: FingerprintRetriever,
    tokenizer: Any,
    prompt_feature_proj: nn.Linear | None,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    max_seq_len = int(config.get("training", {}).get("max_seq_len", 512))
    max_new_tokens = int(config.get("eval", {}).get("max_new_tokens", 512))
    stats_dim = int(config.get("retriever", {}).get("stats_dim", 2))
    feature_dim = int(config.get("model", {}).get("sample_feature_dim", 64))

    static_delta = _static_delta_weight(adapter_manager) if baseline == BASELINE_STATIC else None

    gt_actions: list[str] = []
    gt_severities: list[str] = []
    gt_officials: list[str] = []
    pred_actions: list[str] = []
    pred_severities: list[str] = []
    pred_officials: list[str] = []
    latencies: list[float] = []
    effective_ranks: list[float] = []

    if adapter_manager.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter_manager.device)

    record_iter = records
    if _use_eval_tqdm(config):
        record_iter = tqdm(
            records,
            total=len(records),
            desc=f"Eval [{baseline}][{split}]",
        )

    trace_handle = trace_path.open("w", encoding="utf-8") if trace_path is not None else None
    try:
        for sample_index, record in enumerate(record_iter):
            # Extract ground truth fields from output JSON
            gt_parsed = parse_output_json(str(record.get("output", "")))
            gt_action = str(gt_parsed.get("action", "")).lower().strip() if gt_parsed else ""
            gt_severity = str(gt_parsed.get("severity", "")).lower().strip() if gt_parsed else ""
            gt_official = str(gt_parsed.get("official", "")).lower().strip() if gt_parsed else ""
            gt_actions.append(gt_action)
            gt_severities.append(gt_severity)
            gt_officials.append(gt_official)

            if adapter_manager.device.type == "cuda":
                torch.cuda.synchronize(adapter_manager.device)
            start = time.perf_counter()
            _hyper_out_for_rank: dict[str, Any] | None = None

            with torch.no_grad():
                sample_features: torch.Tensor | None = None
                prompt_text = build_prompt_text(record)

                if baseline != BASELINE_STATIC:
                    sample_features = _encode_prompt_features(
                        prompt_texts=[prompt_text],
                        adapter_manager=adapter_manager,
                        tokenizer=tokenizer,
                        retriever=retriever,
                        prompt_feature_proj=prompt_feature_proj,
                        max_seq_len=max_seq_len,
                    )

                if baseline == BASELINE_STATIC:
                    adapter_manager.inject_delta(delta_weight=static_delta, mode="forward_hook")
                    try:
                        generated_text = _generate_response(
                            record=record,
                            adapter_manager=adapter_manager,
                            tokenizer=tokenizer,
                            max_seq_len=max_seq_len,
                            max_new_tokens=max_new_tokens,
                        )
                    finally:
                        adapter_manager.clear_injection()

                elif baseline == BASELINE_RESIDUAL_NO_RETRIEVER:
                    if sample_features is None:
                        raise RuntimeError("sample_features were not encoded for residual evaluation")
                    retriever_stats = torch.zeros(
                        1, stats_dim,
                        device=adapter_manager.device,
                        dtype=sample_features.dtype,
                    )
                    hyper_out = hyper_residual(sample_features=sample_features, memory_tokens=None)
                    alpha = merge_hypernet(hidden=hyper_out["hidden"], retriever_stats=retriever_stats)
                    delta_weight = adapter_manager.apply_residual(residual_factors=hyper_out, alpha=alpha)
                    adapter_manager.inject_delta(delta_weight=delta_weight, mode="forward_hook")
                    try:
                        generated_text = _generate_response(
                            record=record,
                            adapter_manager=adapter_manager,
                            tokenizer=tokenizer,
                            max_seq_len=max_seq_len,
                            max_new_tokens=max_new_tokens,
                        )
                    finally:
                        adapter_manager.clear_injection()
                    _hyper_out_for_rank = hyper_out

                elif baseline == BASELINE_RESIDUAL_WITH_RETRIEVER:
                    if sample_features is None:
                        raise RuntimeError("sample_features were not encoded for residual evaluation")
                    retrieved = retriever.retrieve(record)
                    memory_tokens_raw = retrieved.get("memory_tokens")
                    memory_tokens = None
                    if isinstance(memory_tokens_raw, torch.Tensor):
                        memory_tokens = memory_tokens_raw.to(
                            device=adapter_manager.device,
                            dtype=sample_features.dtype,
                        ).unsqueeze(0) if memory_tokens_raw.dim() == 2 else memory_tokens_raw.to(
                            device=adapter_manager.device,
                            dtype=sample_features.dtype,
                        )
                    retriever_stats_raw = retrieved.get("retriever_stats")
                    if isinstance(retriever_stats_raw, torch.Tensor):
                        retriever_stats = retriever_stats_raw
                        if retriever_stats.dim() == 1:
                            retriever_stats = retriever_stats.unsqueeze(0)
                        retriever_stats = retriever_stats.to(
                            device=adapter_manager.device,
                            dtype=sample_features.dtype,
                        )
                    else:
                        retriever_stats = torch.zeros(
                            1, stats_dim,
                            device=adapter_manager.device,
                            dtype=sample_features.dtype,
                        )
                    hyper_out = hyper_residual(
                        sample_features=sample_features,
                        memory_tokens=memory_tokens,
                    )
                    alpha = merge_hypernet(
                        hidden=hyper_out["hidden"],
                        retriever_stats=retriever_stats,
                    )
                    delta_weight = adapter_manager.apply_residual(residual_factors=hyper_out, alpha=alpha)
                    adapter_manager.inject_delta(delta_weight=delta_weight, mode="forward_hook")
                    try:
                        generated_text = _generate_response(
                            record=record,
                            adapter_manager=adapter_manager,
                            tokenizer=tokenizer,
                            max_seq_len=max_seq_len,
                            max_new_tokens=max_new_tokens,
                        )
                    finally:
                        adapter_manager.clear_injection()
                    _hyper_out_for_rank = hyper_out
                else:
                    raise ValueError(f"Unsupported baseline: {baseline}")

            if adapter_manager.device.type == "cuda":
                torch.cuda.synchronize(adapter_manager.device)
            latencies.append((time.perf_counter() - start) * 1000.0)

            # Parse generated output for predicted fields
            pred_parsed = parse_output_json(generated_text)
            pred_action = str(pred_parsed.get("action", "")).lower().strip() if pred_parsed else ""
            pred_severity = str(pred_parsed.get("severity", "")).lower().strip() if pred_parsed else ""
            pred_official = str(pred_parsed.get("official", "")).lower().strip() if pred_parsed else ""
            pred_actions.append(pred_action)
            pred_severities.append(pred_severity)
            pred_officials.append(pred_official)

            if _hyper_out_for_rank is not None:
                effective_ranks.append(_sample_effective_rank(_hyper_out_for_rank))
            else:
                effective_ranks.append(0.0)

            parse_status = _classify_parse_status(generated_text, pred_parsed)

            if trace_handle is not None:
                trace_row = {
                    "baseline": baseline,
                    "split": split,
                    "sample_index": sample_index,
                    "sample_id": str(record.get("sample_id", "")),
                    "attack_family": str(record.get("attack_family", "")),
                    "parse_ok": bool(pred_parsed is not None),
                    "parse_status": parse_status,
                    "gt_action": gt_action,
                    "gt_severity": gt_severity,
                    "gt_official": gt_official,
                    "pred_action": pred_action,
                    "pred_severity": pred_severity,
                    "pred_official": pred_official,
                    "generated_text": generated_text,
                }
                trace_handle.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
    finally:
        if trace_handle is not None:
            trace_handle.close()

    metrics = compute_generative_metrics(
        gt_actions=gt_actions,
        gt_severities=gt_severities,
        gt_officials=gt_officials,
        pred_actions=pred_actions,
        pred_severities=pred_severities,
        pred_officials=pred_officials,
    )
    latency_stats = compute_latency_stats(latencies_ms=latencies)

    if adapter_manager.device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(adapter_manager.device) / float(1024**2)
        memory_mode = "cuda_vram"
    else:
        peak_vram_mb = 0.0
        memory_mode = "cpu_only"

    return {
        "baseline": baseline,
        "split": split,
        "sample_count": len(records),
        **metrics,
        **latency_stats,
        "peak_vram_mb": float(peak_vram_mb),
        "effective_rank_mean": float(sum(effective_ranks) / max(len(effective_ranks), 1)),
        "effective_rank_max": float(max(effective_ranks) if effective_ranks else 0.0),
        "ood_gap": 0.0,
        "memory_mode": memory_mode,
    }


def _attach_ood_gap(rows: list[dict[str, Any]]) -> None:
    id_score: dict[str, float] = {}
    ood_score: dict[str, float] = {}
    for row in rows:
        baseline = str(row["baseline"])
        if row["split"] == "test_id":
            id_score[baseline] = float(row.get("exact_match", 0.0))
        elif row["split"] == "test_ood":
            ood_score[baseline] = float(row.get("exact_match", 0.0))

    for row in rows:
        baseline = str(row["baseline"])
        row["ood_gap"] = float(id_score.get(baseline, 0.0) - ood_score.get(baseline, 0.0))


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in TABLE_COLUMNS})


def run_evaluation(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    baselines = resolve_baselines(config.get("eval", {}).get("baselines"))
    tiny_first_n = _eval_tiny_first_n(config)
    checkpoint_path = _resolve_checkpoint(config=config, checkpoint=checkpoint)
    output_root = Path(str(config.get("project", {}).get("output_root", "outputs")))
    artifact_dir = ensure_dir(output_root / "eval" / _experiment_name(config))
    trace_dir = ensure_dir(artifact_dir / "sample_traces")

    adapter_manager = BaseAdapterManager(config)
    adapter_manager.load_base_and_lora(strict=True)

    layer_shapes: dict[str, HyperLayerShape] = {
        layer_name: HyperLayerShape(
            in_features=spec.in_features,
            out_features=spec.out_features,
        )
        for layer_name, spec in adapter_manager.layer_specs.items()
    }
    residual_a1_init_tau = float(config.get("model", {}).get("residual_a1_init_tau", 0.05))
    a_head_init_std_by_layer = adapter_manager.derive_residual_a1_init_std_by_layer(
        tau=residual_a1_init_tau
    )
    hyper_residual = HyperResidualNet(
        config,
        layer_shapes=layer_shapes,
        a_head_init_std_by_layer=a_head_init_std_by_layer,
    ).to(adapter_manager.device)
    merge_hypernet = MergeHyperNet(config, layer_names=list(layer_shapes.keys())).to(adapter_manager.device)

    state = torch.load(checkpoint_path, map_location=adapter_manager.device)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    if "hyper_residual" not in state or "merge_hypernet" not in state:
        raise ValueError(
            f"Checkpoint missing keys 'hyper_residual'/'merge_hypernet': {checkpoint_path}"
        )
    hyper_residual.load_state_dict(state["hyper_residual"], strict=True)
    merge_hypernet.load_state_dict(state["merge_hypernet"], strict=True)
    hyper_residual.eval()
    merge_hypernet.eval()

    tokenizer = _build_tokenizer(config=config, adapter_manager=adapter_manager)
    retriever = FingerprintRetriever.from_config(config)
    prompt_feature_proj_required = (
        (not adapter_manager.is_mock_base)
        and any(baseline != BASELINE_STATIC for baseline in baselines)
    )
    prompt_feature_proj = _build_prompt_feature_proj(
        config=config,
        adapter_manager=adapter_manager,
        checkpoint_state=state,
        checkpoint_path=checkpoint_path,
        required=prompt_feature_proj_required,
    )

    if config.get("data", {}).get("real", {}).get("source_json"):
        manifest = load_real_dataset(config)
    else:
        manifest = generate_synthetic_dataset(config)

    train_dataset = AttackDataset.from_manifest(manifest, split="train")
    retriever.build_index(train_dataset.records)

    rows: list[dict[str, Any]] = []
    sample_trace_paths: dict[str, str] = {}
    for baseline in baselines:
        for split in EVAL_SPLITS:
            dataset = AttackDataset.from_manifest(manifest, split=split)
            records = dataset.records[:tiny_first_n] if tiny_first_n > 0 else dataset.records
            trace_path = (trace_dir / f"{baseline}__{split}.jsonl").resolve()
            sample_trace_paths[f"{baseline}:{split}"] = str(trace_path)
            rows.append(
                _evaluate_split(
                    config=config,
                    baseline=baseline,
                    split=split,
                    records=records,
                    adapter_manager=adapter_manager,
                    hyper_residual=hyper_residual,
                    merge_hypernet=merge_hypernet,
                    retriever=retriever,
                    tokenizer=tokenizer,
                    prompt_feature_proj=prompt_feature_proj,
                    trace_path=trace_path,
                )
            )

    _attach_ood_gap(rows)
    table_text = render_table(rows=rows, columns=TABLE_COLUMNS)

    csv_path = (artifact_dir / "eval_table.csv").resolve()
    json_path = (artifact_dir / "eval_summary.json").resolve()

    _save_csv(csv_path, rows)
    result: dict[str, Any] = {
        "status": "eval_ok",
        "checkpoint_used": str(checkpoint_path),
        "tiny_first_n": tiny_first_n,
        "rows": rows,
        "table_text": table_text,
        "artifacts": {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "sample_trace_dir": str(trace_dir.resolve()),
            "sample_trace_paths": sample_trace_paths,
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
