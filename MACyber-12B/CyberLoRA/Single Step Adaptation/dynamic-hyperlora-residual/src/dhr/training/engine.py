from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback.
    tqdm = None

from dhr.data.dataset import AttackDataset
from dhr.data.collator import simple_collator
from dhr.data.convert_real import load_real_dataset
from dhr.data.synthetic import generate_synthetic_dataset
from dhr.modeling.base_adapter import BaseAdapterManager
from dhr.modeling.hyper_residual import HyperLayerShape, HyperResidualNet
from dhr.modeling.merge_hypernet import MergeHyperNet
from dhr.modeling.prompt_feature_projection import (
    build_prompt_feature_projection,
    prompt_feature_projection_metadata,
    resolve_prompt_feature_hidden_size,
)
from dhr.modeling.rank_controller import RankController
from dhr.modeling.retriever import FingerprintRetriever
from dhr.training.losses import compute_total_loss
from dhr.training.metrics_artifacts import save_training_artifacts
from dhr.utils.hf_kwargs import hf_extra_kwargs_from_model_cfg
from dhr.utils.model_forward import last_hidden_state_from_base_model
from dhr.utils.io import ensure_dir
from dhr.utils.logging import setup_logger
from dhr.utils.prompt_features import build_prompt_text, pack_hidden_features


def _linalg_safe_rank(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.dtype in {torch.float32, torch.float64}:
        return torch.linalg.matrix_rank(matrix)
    return torch.linalg.matrix_rank(matrix.to(dtype=torch.float32))


def _low_rank_core_matrix(a_factor: torch.Tensor, b_factor: torch.Tensor) -> torch.Tensor:
    """
    Build the exact small core matrix for W=A@B:
    svd/rank(W) equals svd/rank(core), where core = R_a @ R_b^T
    from reduced QR of A and B^T.
    """
    a_safe = a_factor if a_factor.dtype in {torch.float32, torch.float64} else a_factor.to(torch.float32)
    b_safe = b_factor if b_factor.dtype in {torch.float32, torch.float64} else b_factor.to(torch.float32)
    _, r_a = torch.linalg.qr(a_safe, mode="reduced")
    _, r_bt = torch.linalg.qr(b_safe.transpose(-2, -1), mode="reduced")
    return r_a @ r_bt.transpose(-2, -1)


@dataclass
class EngineSummary:
    steps: int
    loss: float
    status: str
    m2_payload: dict[str, Any]


class TrainingEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_cfg = config.get("model", {})
        self.training_cfg = config.get("training", {})
        self.project_cfg = config.get("project", {})
        self.retriever_cfg = config.get("retriever", {})
        self.logger = setup_logger("dhr.training")

        self.device = torch.device(self.model_cfg.get("device", "cpu"))
        self.base_input_device = self.device
        self.feature_dim = int(self.model_cfg.get("sample_feature_dim", 64))
        self.stats_dim = int(self.retriever_cfg.get("stats_dim", 2))
        self.max_seq_len = int(self.training_cfg.get("max_seq_len", 256))
        self.min_output_tokens = 64
        self.lambda_orth = float(self.training_cfg.get("lambda_orth", 1e-3))
        self.lambda_rank = float(self.training_cfg.get("lambda_rank", 5e-4))
        self.grad_accum_steps = max(int(self.training_cfg.get("grad_accum_steps", 1)), 1)
        self.grad_clip = float(self.training_cfg.get("grad_clip", 1.0))
        self.epochs = max(int(self.training_cfg.get("epochs", 1)), 1)
        self.prune_every_steps = max(int(self.training_cfg.get("prune_every_steps", 0)), 0)
        self.precision = str(self.training_cfg.get("precision", "fp32")).lower()

        self.global_step = 0
        self.best_val_loss = float("inf")

        self.adapter_manager: BaseAdapterManager | None = None
        self.hyper_residual: HyperResidualNet | None = None
        self.merge_hypernet: MergeHyperNet | None = None
        self.retriever: FingerprintRetriever | None = None
        self.rank_controller: RankController | None = None
        self.tokenizer: Any = None
        # Projects LLM hidden states → HyperNet feature_dim space
        self.prompt_feature_proj: nn.Linear | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.grad_scaler: torch.cuda.amp.GradScaler | None = None
        self.checkpoint_dir: Path | None = None

    def _use_tqdm(self) -> bool:
        return bool(self.training_cfg.get("use_tqdm", True)) and tqdm is not None

    def dry_run(self) -> EngineSummary:
        epochs = int(self.config["training"]["epochs"])

        adapter_manager = BaseAdapterManager(self.config)
        adapter_manager.load_base_and_lora(strict=True)
        device = adapter_manager.device

        layer_shapes: dict[str, HyperLayerShape] = {
            layer_name: HyperLayerShape(
                in_features=spec.in_features,
                out_features=spec.out_features,
            )
            for layer_name, spec in adapter_manager.layer_specs.items()
        }
        residual_a1_init_tau = float(self.model_cfg.get("residual_a1_init_tau", 0.05))
        a_head_init_std_by_layer = adapter_manager.derive_residual_a1_init_std_by_layer(
            tau=residual_a1_init_tau
        )
        hyper_residual = HyperResidualNet(
            self.config,
            layer_shapes=layer_shapes,
            a_head_init_std_by_layer=a_head_init_std_by_layer,
        ).to(device)
        merge_hypernet = MergeHyperNet(self.config, layer_names=list(layer_shapes.keys())).to(device)
        retriever = FingerprintRetriever.from_config(self.config)

        if self.config.get("data", {}).get("real", {}).get("source_json"):
            manifest = load_real_dataset(self.config)
        else:
            manifest = generate_synthetic_dataset(self.config)
        train_dataset = AttackDataset.from_manifest(manifest, split="train")
        probe_dataset = AttackDataset.from_manifest(manifest, split="val")
        if len(probe_dataset) == 0:
            probe_dataset = train_dataset
        if len(train_dataset) == 0:
            raise RuntimeError("synthetic train split is empty")

        retriever.build_index(train_dataset.records)
        probe_sample = probe_dataset[0]

        # Use hash-based query embedding as a probe for dry_run
        retrieved = retriever.retrieve(probe_sample)
        sample_features = retrieved["query_embedding"].unsqueeze(0).to(device)
        memory_tokens = retrieved.get("memory_tokens")
        if isinstance(memory_tokens, torch.Tensor):
            memory_tokens = memory_tokens.to(device)
        retriever_stats = retrieved.get("retriever_stats")
        if isinstance(retriever_stats, torch.Tensor):
            retriever_stats = retriever_stats.unsqueeze(0).to(device)

        hyper_out = hyper_residual(sample_features=sample_features, memory_tokens=memory_tokens)
        alpha = merge_hypernet(hidden=hyper_out["hidden"], retriever_stats=retriever_stats)
        delta_weight = adapter_manager.apply_residual(residual_factors=hyper_out, alpha=alpha)
        injected_layer_count = adapter_manager.inject_delta(delta_weight=delta_weight, mode="forward_hook")

        mock_forward_ok = False
        if adapter_manager.is_mock_base and adapter_manager.base_model is not None:
            _ = adapter_manager.base_model(sample_features)
            mock_forward_ok = True

        effective_rank: dict[str, int] = {}
        for layer_name in delta_weight:
            residual_key = layer_name
            if residual_key not in hyper_out["lora_A1"] or residual_key not in hyper_out["lora_B1"]:
                residual_key = adapter_manager._target_group_by_module.get(layer_name, layer_name)
            a_factor = hyper_out["lora_A1"][residual_key]
            b_factor = hyper_out["lora_B1"][residual_key]
            core = _low_rank_core_matrix(a_factor, b_factor)
            rank_tensor = _linalg_safe_rank(core)
            rank = int(rank_tensor.max().item() if rank_tensor.dim() > 0 else rank_tensor.item())
            effective_rank[layer_name] = rank

        m2_payload = {
            "a1_shapes": {
                layer_name: list(tensor.shape)
                for layer_name, tensor in hyper_out["lora_A1"].items()
            },
            "b1_shapes": {
                layer_name: list(tensor.shape)
                for layer_name, tensor in hyper_out["lora_B1"].items()
            },
            "alpha": {
                layer_name: float(layer_alpha.mean().item())
                if isinstance(layer_alpha, torch.Tensor) and layer_alpha.dim() > 0
                else float(layer_alpha.item() if isinstance(layer_alpha, torch.Tensor) else layer_alpha)
                for layer_name, layer_alpha in alpha.items()
            },
            "delta_shapes": {
                layer_name: list(tensor.shape)
                for layer_name, tensor in delta_weight.items()
            },
            "effective_rank": effective_rank,
            "injected_layer_count": injected_layer_count,
            "mock_forward_ok": mock_forward_ok,
            "m3_manifest_counts": manifest["counts"],
            "m3_retriever": {
                "index_size": retriever.index_size,
                "fallback": bool(retrieved.get("fallback", True)),
                "topk_size": len(retrieved.get("topk_ids", [])),
            },
        }

        adapter_manager.clear_injection()
        return EngineSummary(
            steps=epochs,
            loss=0.0,
            status="dry_run_ok",
            m2_payload=m2_payload,
        )

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader, dict[str, Any], AttackDataset]:
        if self.config.get("data", {}).get("real", {}).get("source_json"):
            manifest = load_real_dataset(self.config)
        else:
            manifest = generate_synthetic_dataset(self.config)
        train_dataset = AttackDataset.from_manifest(manifest, split="train")
        val_dataset = AttackDataset.from_manifest(manifest, split="val")
        if len(train_dataset) == 0:
            raise RuntimeError("synthetic train split is empty")
        if len(val_dataset) == 0:
            val_dataset = train_dataset

        batch_size = max(int(self.training_cfg.get("batch_size", 8)), 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=simple_collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=simple_collator,
        )
        return train_loader, val_loader, manifest, train_dataset

    def _initialize_modeling(self, strict: bool) -> None:
        self.adapter_manager = BaseAdapterManager(self.config)
        self.adapter_manager.load_base_and_lora(strict=strict)
        self.device = self.adapter_manager.device
        self.base_input_device = self.adapter_manager.base_model_input_device()

        layer_shapes: dict[str, HyperLayerShape] = {
            layer_name: HyperLayerShape(
                in_features=spec.in_features,
                out_features=spec.out_features,
            )
            for layer_name, spec in self.adapter_manager.layer_specs.items()
        }
        residual_a1_init_tau = float(self.model_cfg.get("residual_a1_init_tau", 0.05))
        a_head_init_std_by_layer = self.adapter_manager.derive_residual_a1_init_std_by_layer(
            tau=residual_a1_init_tau
        )
        self.hyper_residual = HyperResidualNet(
            self.config,
            layer_shapes=layer_shapes,
            a_head_init_std_by_layer=a_head_init_std_by_layer,
        ).to(self.device)
        self.merge_hypernet = MergeHyperNet(
            self.config,
            layer_names=list(layer_shapes.keys()),
        ).to(self.device)
        self.rank_controller = RankController(
            target_rank=int(self.training_cfg.get("target_rank", 8)),
            tol=int(self.training_cfg.get("rank_tol", 2)),
        )
        self.retriever = FingerprintRetriever.from_config(self.config)

        if not self.adapter_manager.is_mock_base:
            try:
                from transformers import AutoTokenizer
            except Exception as exc:  # pragma: no cover - import environment dependent.
                raise RuntimeError("transformers is required for real backbone training") from exc

            base_model_id = str(self.model_cfg.get("base_model", ""))
            if not base_model_id:
                raise ValueError("model.base_model must be provided when not using mock base")
            tok_kw: dict[str, Any] = {
                "local_files_only": bool(self.model_cfg.get("local_files_only", False)),
            }
            tok_kw.update(hf_extra_kwargs_from_model_cfg(self.model_cfg))
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, **tok_kw)

            # Build prompt feature projection: LLM hidden_size → HyperNet feature_dim
            base_model = self.adapter_manager.base_model
            hidden_size = resolve_prompt_feature_hidden_size(
                base_model,
                fallback_hidden_size=self.feature_dim,
            )
            self.prompt_feature_proj = build_prompt_feature_projection(
                hidden_size=hidden_size,
                feature_dim=self.feature_dim,
                device=self.device,
                dtype=torch.float32,
            )

    @staticmethod
    def _build_linear_warmup_decay(
        warmup_steps: int,
        total_steps: int,
    ):
        def lr_lambda(current_step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(warmup_steps, 1))
            remain_steps = max(total_steps - warmup_steps, 1)
            remain = max(total_steps - current_step, 0)
            return float(remain) / float(remain_steps)

        return lr_lambda

    def _setup_optim(self, train_loader_len: int) -> None:
        if self.hyper_residual is None or self.merge_hypernet is None:
            raise RuntimeError("modeling modules are not initialized")

        trainable_params = (
            list(self.hyper_residual.parameters())
            + list(self.merge_hypernet.parameters())
        )
        if self.prompt_feature_proj is not None:
            trainable_params += list(self.prompt_feature_proj.parameters())

        lr = float(self.training_cfg.get("lr", 3e-4))
        wd = float(self.training_cfg.get("weight_decay", 0.01))
        self.optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd)

        steps_per_epoch = max((train_loader_len + self.grad_accum_steps - 1) // self.grad_accum_steps, 1)
        total_steps = max(steps_per_epoch * self.epochs, 1)
        warmup_ratio = float(self.training_cfg.get("warmup_ratio", 0.05))
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._build_linear_warmup_decay(
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            ),
        )

        use_scaler = self.device.type == "cuda" and self.precision in {"fp16", "float16"}
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        if self.precision in {"bf16", "bfloat16"}:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.precision in {"fp16", "float16"}:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _encode_prompt_features(self, prompt_texts: list[str]) -> torch.Tensor:
        """Run no-grad LLM forward on prompt texts, pack hidden states, project to feature_dim.

        Returns: [B, feature_dim] float32 tensor on self.device.
        Falls back to hash-based retriever encoding for mock base.
        """
        if self.adapter_manager is None:
            raise RuntimeError("adapter_manager is not initialized")

        if self.adapter_manager.is_mock_base or self.tokenizer is None:
            # Mock path: use hash vectorization via retriever (fallback)
            if self.retriever is None:
                raise RuntimeError("retriever is not initialized")
            vecs = [
                self.retriever.encode_query(text).to(dtype=torch.float32)
                for text in prompt_texts
            ]
            return torch.stack(vecs, dim=0).to(self.device)

        if self.prompt_feature_proj is None:
            raise RuntimeError("prompt_feature_proj is not initialized")

        tokenized = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.base_input_device) for k, v in tokenized.items()}

        with torch.no_grad():
            hidden = last_hidden_state_from_base_model(
                self.adapter_manager.base_model,
                **tokenized,
            )

        packed = pack_hidden_features(hidden=hidden, attention_mask=tokenized.get("attention_mask"))
        packed_f32 = packed.to(dtype=torch.float32, device=self.device)
        return self.prompt_feature_proj(packed_f32)  # [B, feature_dim]

    def _pack_conditioning(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build sample_features (from LLM hidden states), memory_tokens, retriever_stats.

        sample_features: no-grad LLM forward on prompt (instruction+input), projected to feature_dim.
        memory_tokens: retrieved from the hash-based index (unchanged).
        retriever_stats: similarity statistics from retriever (unchanged).
        """
        if self.retriever is None:
            raise RuntimeError("retriever is not initialized")

        # Build prompt texts for LLM encoding
        prompt_texts = [build_prompt_text(record) for record in records]

        # Encode prompts via LLM (no-grad) → [B, feature_dim]
        sample_features = self._encode_prompt_features(prompt_texts)

        # Memory tokens and stats still come from hash-based retriever
        memory_list: list[torch.Tensor] = []
        stats_list: list[torch.Tensor] = []

        for record in records:
            retrieved = self.retriever.retrieve(record)

            memory_tokens = retrieved.get("memory_tokens")
            if isinstance(memory_tokens, torch.Tensor):
                memory_list.append(memory_tokens.to(dtype=torch.float32))
            else:
                # Fallback: use query embedding as a single memory token
                memory_list.append(retrieved["query_embedding"].unsqueeze(0).to(dtype=torch.float32))

            stats = retrieved.get("retriever_stats")
            if isinstance(stats, torch.Tensor):
                stats_list.append(stats.to(dtype=torch.float32))
            else:
                stats_list.append(torch.zeros(self.stats_dim, dtype=torch.float32))

        max_tokens = max(mem.shape[0] for mem in memory_list)
        memory_tokens_batch = torch.zeros(
            len(memory_list),
            max_tokens,
            self.feature_dim,
            device=self.device,
            dtype=sample_features.dtype,
        )
        for idx, memory in enumerate(memory_list):
            token_count = memory.shape[0]
            memory_tokens_batch[idx, :token_count, :] = memory.to(
                device=self.device,
                dtype=sample_features.dtype,
            )

        retriever_stats = torch.stack(stats_list, dim=0).to(
            device=self.device,
            dtype=sample_features.dtype,
        )
        return sample_features, memory_tokens_batch, retriever_stats

    def _build_lm_inputs(
        self,
        records: list[dict[str, Any]],
        phase: str | None = None,
        epoch_idx: int | None = None,
        batch_idx: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Tokenize full sequences and build masked labels for LM loss.

        Uses apply_chat_template if the tokenizer supports it.
        Prompt tokens (instruction+input) are masked to -100 in labels.

        Returns:
            model_inputs: tokenized dict on base_input_device
            labels: [B, seq_len] long tensor with -100 for prompt positions
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not initialized")

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and (
            self.tokenizer.chat_template is not None
        )

        for sample_idx, record in enumerate(records):
            instruction = str(record.get("instruction", ""))
            inp = str(record.get("input", ""))
            output = str(record.get("output", ""))

            if has_chat_template:
                # Build messages for chat template
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": inp},
                ]
                # Tokenize prompt only to find the boundary
                _prompt_out = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )  # tensor [1, prompt_len] or BatchEncoding
                # Some tokenizers return BatchEncoding instead of a raw tensor
                prompt_ids = _prompt_out["input_ids"] if hasattr(_prompt_out, "keys") else _prompt_out

                # Full sequence: prompt + response
                messages_full = messages + [{"role": "assistant", "content": output}]
                _full_out = self.tokenizer.apply_chat_template(
                    messages_full,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )  # tensor [1, full_len] or BatchEncoding
                full_ids = _full_out["input_ids"] if hasattr(_full_out, "keys") else _full_out
            else:
                # Plain concatenation fallback
                prompt_text = f"{instruction}\n\n{inp}\n\n" if inp else f"{instruction}\n\n"
                prompt_enc = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                prompt_ids = prompt_enc["input_ids"]

                full_text = prompt_text + output
                full_enc = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                full_ids = full_enc["input_ids"]

            prompt_ids = prompt_ids[0]
            full_ids = full_ids[0]  # [full_len]
            prompt_len = min(int(prompt_ids.shape[0]), int(full_ids.shape[0]))
            output_ids = full_ids[prompt_len:]
            output_len = int(output_ids.shape[0])
            full_len = int(full_ids.shape[0])

            seq_ids = full_ids
            if full_len > self.max_seq_len:
                if output_len >= self.max_seq_len:
                    # Output segment alone exceeds max length: keep output prefix only.
                    kept_prompt_ids = prompt_ids[:0]
                    kept_output_ids = output_ids[: self.max_seq_len]
                else:
                    # Keep all output tokens, then fit prompt tail into remaining budget.
                    keep_prompt_len = max(self.max_seq_len - output_len, 0)
                    kept_prompt_ids = prompt_ids[-keep_prompt_len:] if keep_prompt_len > 0 else prompt_ids[:0]
                    kept_output_ids = output_ids

                seq_ids = torch.cat((kept_prompt_ids, kept_output_ids), dim=0)
                new_prompt_len = int(kept_prompt_ids.shape[0])
                if new_prompt_len < prompt_len:
                    self.logger.warning(
                        (
                            "Prompt truncated to preserve output budget: phase=%s epoch=%s batch=%s sample=%d "
                            "(prompt=%d output=%d full=%d -> kept_prompt=%d kept_output=%d final=%d "
                            "reserved_output=%d output_truncated=%s)"
                        ),
                        phase or "unknown",
                        epoch_idx if epoch_idx is not None else -1,
                        batch_idx if batch_idx is not None else -1,
                        sample_idx,
                        prompt_len,
                        output_len,
                        full_len,
                        new_prompt_len,
                        int(kept_output_ids.shape[0]),
                        int(seq_ids.shape[0]),
                        min(self.min_output_tokens, output_len),
                        bool(int(kept_output_ids.shape[0]) < output_len),
                    )
                prompt_len = new_prompt_len

            lbl = seq_ids.clone()
            lbl[:prompt_len] = -100  # mask prompt tokens

            input_ids_list.append(seq_ids)
            labels_list.append(lbl)

        # Pad to max length in batch
        max_len = max(t.shape[0] for t in input_ids_list)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        batch_input_ids = torch.full(
            (len(input_ids_list), max_len),
            fill_value=pad_id,
            dtype=torch.long,
        )
        batch_attention_mask = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
        batch_labels = torch.full(
            (len(labels_list), max_len),
            fill_value=-100,
            dtype=torch.long,
        )

        for i, (ids, lbl) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = ids.shape[0]
            batch_input_ids[i, :seq_len] = ids
            batch_attention_mask[i, :seq_len] = 1
            batch_labels[i, :seq_len] = lbl

        model_inputs = {
            "input_ids": batch_input_ids.to(self.base_input_device),
            "attention_mask": batch_attention_mask.to(self.base_input_device),
        }
        return model_inputs, batch_labels.to(self.device)

    def _forward_lm(
        self,
        records: list[dict[str, Any]],
        phase: str | None = None,
        epoch_idx: int | None = None,
        batch_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for LM loss with injected LoRA delta.

        Returns:
            logits: [B, seq_len, vocab_size]
            labels: [B, seq_len] with -100 for prompt/padding positions
        """
        if self.adapter_manager is None or self.adapter_manager.base_model is None:
            raise RuntimeError("base model is not initialized")

        if self.adapter_manager.is_mock_base:
            raise RuntimeError("_forward_lm is not supported for mock base")

        model_inputs, labels = self._build_lm_inputs(
            records,
            phase=phase,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
        )
        output = self.adapter_manager.base_model(**model_inputs)

        if isinstance(output, dict):
            logits = output.get("logits")
        else:
            logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError("base model output missing 'logits'")

        return logits.to(self.device), labels

    @staticmethod
    def _mean_effective_rank(hyper_out: dict[str, Any]) -> float:
        lora_a1 = hyper_out["lora_A1"]
        lora_b1 = hyper_out["lora_B1"]
        ranks = []
        for layer_name in lora_a1:
            a_factor = lora_a1[layer_name]
            b_factor = lora_b1[layer_name]
            core = _low_rank_core_matrix(a_factor, b_factor)
            rank_tensor = _linalg_safe_rank(core)
            if rank_tensor.dim() == 0:
                rank_value = float(rank_tensor.item())
            else:
                rank_value = float(rank_tensor.max().item())
            ranks.append(float(rank_value))
        return float(sum(ranks) / max(len(ranks), 1))

    def _run_batch(
        self,
        records: list[dict[str, Any]],
        training: bool,
        epoch_idx: int | None = None,
        batch_idx: int | None = None,
    ) -> dict[str, Any]:
        if (
            self.adapter_manager is None
            or self.hyper_residual is None
            or self.merge_hypernet is None
            or self.rank_controller is None
        ):
            raise RuntimeError("modeling stack is not initialized")

        sample_features, memory_tokens, retriever_stats = self._pack_conditioning(records)

        with self._autocast_context():
            hyper_out = self.hyper_residual(
                sample_features=sample_features,
                memory_tokens=memory_tokens,
            )
            if (
                training
                and self.prune_every_steps > 0
                and self.global_step > 0
                and self.global_step % self.prune_every_steps == 0
            ):
                with torch.no_grad():
                    hyper_out = self.rank_controller.prune(hyper_out)
            alpha = self.merge_hypernet(
                hidden=hyper_out["hidden"],
                retriever_stats=retriever_stats,
            )
            delta_weight = self.adapter_manager.apply_residual(
                residual_factors=hyper_out,
                alpha=alpha,
            )
            self.adapter_manager.inject_delta(delta_weight=delta_weight, mode="forward_hook")
            try:
                logits, labels = self._forward_lm(
                    records=records,
                    phase="train" if training else "val",
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                )
            finally:
                self.adapter_manager.clear_injection()

            loss_payload = compute_total_loss(
                logits=logits,
                labels=labels,
                residual_factors=hyper_out,
                lambda_orth=self.lambda_orth,
                lambda_rank=self.lambda_rank,
            )

        return {
            "loss": loss_payload,
            "effective_rank": self._mean_effective_rank(hyper_out),
            "batch_size": len(records),
        }

    def train_epoch(self, train_loader: DataLoader, epoch_idx: int) -> dict[str, Any]:
        if self.optimizer is None or self.scheduler is None or self.grad_scaler is None:
            raise RuntimeError("optimizer/scheduler are not initialized")
        if self.hyper_residual is None or self.merge_hypernet is None:
            raise RuntimeError("hyper modules are not initialized")

        self.hyper_residual.train()
        self.merge_hypernet.train()
        if self.prompt_feature_proj is not None:
            self.prompt_feature_proj.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_task = 0.0
        total_orth = 0.0
        total_rank = 0.0
        total_loss = 0.0
        total_effective_rank = 0.0
        batch_count = 0
        step_metrics: list[dict[str, float | int]] = []
        pending_task = 0.0
        pending_orth = 0.0
        pending_rank = 0.0
        pending_total = 0.0
        pending_effective_rank = 0.0
        pending_batches = 0

        progress = None
        train_iter = enumerate(train_loader, start=1)
        if self._use_tqdm():
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch_idx}/{self.epochs} [train]",
                dynamic_ncols=True,
                leave=False,
                file=sys.stdout,
            )
            train_iter = enumerate(progress, start=1)

        try:
            for batch_idx, batch in train_iter:
                records = list(batch["batch"])
                batch_out = self._run_batch(
                    records=records,
                    training=True,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                )
                loss_payload = batch_out["loss"]
                if not torch.isfinite(loss_payload["total"]).item():
                    self.logger.warning(
                        "Skip non-finite loss at epoch=%d batch=%d (loss=%s)",
                        epoch_idx,
                        batch_idx,
                        float(loss_payload["total"].detach().cpu().item()),
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    pending_task = 0.0
                    pending_orth = 0.0
                    pending_rank = 0.0
                    pending_total = 0.0
                    pending_effective_rank = 0.0
                    pending_batches = 0
                    continue
                loss_for_backward = loss_payload["total"] / float(self.grad_accum_steps)

                total_task += float(loss_payload["task"].detach().item())
                total_orth += float(loss_payload["orth"].detach().item())
                total_rank += float(loss_payload["rank"].detach().item())
                total_loss += float(loss_payload["total"].detach().item())
                total_effective_rank += float(batch_out["effective_rank"])
                batch_count += 1
                pending_task += float(loss_payload["task"].detach().item())
                pending_orth += float(loss_payload["orth"].detach().item())
                pending_rank += float(loss_payload["rank"].detach().item())
                pending_total += float(loss_payload["total"].detach().item())
                pending_effective_rank += float(batch_out["effective_rank"])
                pending_batches += 1

                if self.grad_scaler.is_enabled():
                    self.grad_scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = (batch_idx % self.grad_accum_steps == 0) or (batch_idx == len(train_loader))
                if should_step:
                    all_params = (
                        list(self.hyper_residual.parameters())
                        + list(self.merge_hypernet.parameters())
                    )
                    if self.prompt_feature_proj is not None:
                        all_params += list(self.prompt_feature_proj.parameters())

                    if self.grad_scaler.is_enabled():
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if pending_batches > 0:
                        current_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                        step_metrics.append(
                            {
                                "epoch": epoch_idx,
                                "batch_idx": batch_idx,
                                "global_step": self.global_step,
                                "lr": current_lr,
                                "step_task_loss": pending_task / pending_batches,
                                "step_orth_loss": pending_orth / pending_batches,
                                "step_rank_loss": pending_rank / pending_batches,
                                "step_total_loss": pending_total / pending_batches,
                                "step_effective_rank": pending_effective_rank / pending_batches,
                                "micro_batch_count": pending_batches,
                            }
                        )
                        pending_task = 0.0
                        pending_orth = 0.0
                        pending_rank = 0.0
                        pending_total = 0.0
                        pending_effective_rank = 0.0
                        pending_batches = 0

                if progress is not None:
                    running_loss = total_loss / max(batch_count, 1)
                    lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                    progress.set_postfix(loss=f"{running_loss:.4f}", lr=f"{lr:.2e}", step=self.global_step)
        finally:
            if progress is not None:
                progress.close()

        denom = max(batch_count, 1)
        metrics = {
            "train_task_loss": total_task / denom,
            "train_orth_loss": total_orth / denom,
            "train_rank_loss": total_rank / denom,
            "train_loss": total_loss / denom,
            "train_effective_rank": total_effective_rank / denom,
            "step_count": len(step_metrics),
            "step_metrics": step_metrics,
        }
        metrics_for_log = dict(metrics)
        metrics_for_log.pop("step_metrics", None)
        self.logger.info("Epoch %d train metrics: %s", epoch_idx, metrics_for_log)
        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch_idx: int) -> dict[str, float]:
        if self.hyper_residual is None or self.merge_hypernet is None:
            raise RuntimeError("hyper modules are not initialized")
        self.hyper_residual.eval()
        self.merge_hypernet.eval()
        if self.prompt_feature_proj is not None:
            self.prompt_feature_proj.eval()

        total_task = 0.0
        total_orth = 0.0
        total_rank = 0.0
        total_loss = 0.0
        total_effective_rank = 0.0
        batch_count = 0

        progress = None
        val_iter = enumerate(val_loader, start=1)
        if self._use_tqdm():
            progress = tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {epoch_idx}/{self.epochs} [val]",
                dynamic_ncols=True,
                leave=False,
                file=sys.stdout,
            )
            val_iter = enumerate(progress, start=1)

        try:
            for batch_idx, batch in val_iter:
                records = list(batch["batch"])
                batch_out = self._run_batch(
                    records=records,
                    training=False,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                )
                loss_payload = batch_out["loss"]

                total_task += float(loss_payload["task"].detach().item())
                total_orth += float(loss_payload["orth"].detach().item())
                total_rank += float(loss_payload["rank"].detach().item())
                total_loss += float(loss_payload["total"].detach().item())
                total_effective_rank += float(batch_out["effective_rank"])
                batch_count += 1

                if progress is not None:
                    running_loss = total_loss / max(batch_count, 1)
                    progress.set_postfix(loss=f"{running_loss:.4f}")
        finally:
            if progress is not None:
                progress.close()

        denom = max(batch_count, 1)
        metrics = {
            "val_task_loss": total_task / denom,
            "val_orth_loss": total_orth / denom,
            "val_rank_loss": total_rank / denom,
            "val_loss": total_loss / denom,
            "val_effective_rank": total_effective_rank / denom,
        }
        self.logger.info("Epoch %d val metrics: %s", epoch_idx, metrics)
        return metrics

    def _checkpoint_payload(self, epoch_idx: int) -> dict[str, Any]:
        if (
            self.hyper_residual is None
            or self.merge_hypernet is None
            or self.optimizer is None
            or self.scheduler is None
        ):
            raise RuntimeError("checkpoint payload requested before initialization")
        payload: dict[str, Any] = {
            "hyper_residual": self.hyper_residual.state_dict(),
            "merge_hypernet": self.merge_hypernet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch_idx,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config_snapshot": self.config,
        }
        if self.prompt_feature_proj is not None:
            payload["prompt_feature_proj"] = self.prompt_feature_proj.state_dict()
            base_model = (
                self.adapter_manager.base_model
                if self.adapter_manager is not None
                else None
            )
            hidden_size = resolve_prompt_feature_hidden_size(
                base_model,
                fallback_hidden_size=self.feature_dim,
            )
            payload.update(prompt_feature_projection_metadata(hidden_size=hidden_size))
        return payload

    def save_checkpoint(self, checkpoint_type: str, epoch_idx: int) -> str:
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint dir is not prepared")
        payload = self._checkpoint_payload(epoch_idx=epoch_idx)
        ckpt_path = self.checkpoint_dir / f"{checkpoint_type}.pt"
        torch.save(payload, ckpt_path)
        return str(ckpt_path)

    def run(self) -> dict[str, Any]:
        train_loader, val_loader, manifest, train_dataset = self._build_dataloaders()
        self._initialize_modeling(strict=True)
        self._setup_optim(train_loader_len=len(train_loader))

        if self.retriever is None:
            raise RuntimeError("retriever not initialized")
        self.retriever.build_index(train_dataset.records)

        exp_name = str(self.config.get("experiment", {}).get("name", self.project_cfg.get("name", "default")))
        ckpt_root = Path(self.project_cfg.get("checkpoint_root", "checkpoints"))
        output_root = Path(self.project_cfg.get("output_root", "outputs"))
        self.checkpoint_dir = ensure_dir(ckpt_root / exp_name)
        training_artifact_dir = ensure_dir(output_root / exp_name / "train_metrics")

        best_ckpt = ""
        last_ckpt = ""
        latest_train: dict[str, float] = {}
        latest_val: dict[str, float] = {}
        epoch_history: list[dict[str, float]] = []
        step_history: list[dict[str, float | int]] = []
        training_artifacts: dict[str, str] = {}

        for epoch_idx in range(1, self.epochs + 1):
            latest_train_payload = self.train_epoch(train_loader=train_loader, epoch_idx=epoch_idx)
            latest_step_metrics = latest_train_payload.pop("step_metrics", [])
            if isinstance(latest_step_metrics, list):
                step_history.extend(latest_step_metrics)
            latest_train = latest_train_payload
            latest_val = self.validate(val_loader=val_loader, epoch_idx=epoch_idx)

            current_val_loss = float(latest_val["val_loss"])
            last_ckpt = self.save_checkpoint(checkpoint_type="last", epoch_idx=epoch_idx)
            if current_val_loss <= self.best_val_loss:
                self.best_val_loss = current_val_loss
                best_ckpt = self.save_checkpoint(checkpoint_type="best", epoch_idx=epoch_idx)
            self.logger.info(
                "Epoch %d finished | train_loss=%.6f | val_loss=%.6f | best_val=%.6f",
                epoch_idx,
                latest_train["train_loss"],
                current_val_loss,
                self.best_val_loss,
            )

            lr = 0.0
            if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            epoch_metrics = {
                "epoch": epoch_idx,
                "global_step": self.global_step,
                "lr": lr,
                **latest_train,
                **latest_val,
            }
            epoch_history.append(epoch_metrics)
            training_artifacts = save_training_artifacts(
                artifact_dir=training_artifact_dir,
                epoch_history=epoch_history,
                step_history=step_history,
            )

        if not best_ckpt:
            best_ckpt = last_ckpt

        if training_artifacts:
            self.logger.info("Training metric artifacts saved: %s", training_artifacts)

        return {
            "status": "train_ok",
            "steps": self.global_step,
            "train_loss": float(latest_train.get("train_loss", 0.0)),
            "val_loss": float(latest_val.get("val_loss", 0.0)),
            "train_metrics": latest_train,
            "val_metrics": latest_val,
            "best_checkpoint": best_ckpt,
            "last_checkpoint": last_ckpt,
            "manifest_counts": manifest.get("counts", {}),
            "training_artifacts": training_artifacts,
        }
