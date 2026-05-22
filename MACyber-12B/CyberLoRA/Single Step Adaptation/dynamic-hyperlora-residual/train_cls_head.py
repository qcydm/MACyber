from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from dhr.data.convert_real import load_real_dataset
from dhr.data.dataset import AttackDataset
from dhr.data.synthetic import generate_synthetic_dataset
from dhr.modeling.base_adapter import BaseAdapterManager
from dhr.modeling.cls_head import LinearClsHead
from dhr.modeling.lora_ops import compute_lora_weight
from dhr.utils.config import load_config
from dhr.utils.logging import setup_logger
from dhr.utils.model_forward import last_hidden_state_from_base_model
from dhr.utils.seed import set_seed


def _get_hidden_dim(base_model: torch.nn.Module) -> int:
    cfg = getattr(base_model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)
    raise RuntimeError(
        "Cannot infer hidden_dim from base_model.config. "
        "Add model.cls_head_hidden_dim to config as fallback."
    )


def _static_delta(adapter_manager: BaseAdapterManager) -> dict[str, torch.Tensor]:
    base_factors = adapter_manager.get_base_lora_factors()
    return {
        layer_name: compute_lora_weight(factors["A0"], factors["B0"])
        for layer_name, factors in base_factors.items()
    }


def _tokenize(record: dict, tokenizer, max_seq_len: int, input_device: torch.device):
    text = f"{record['instruction']}\n{record['input']}".strip()
    tokenized = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return {k: v.to(input_device) for k, v in tokenized.items()}


def run_cls_head_training(config: dict, output: str | Path) -> dict:
    logger = setup_logger("dhr.train_cls_head")
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    lr = float(training_cfg.get("lr", 3e-4))
    epochs = max(int(training_cfg.get("epochs", 8)), 1)
    batch_size = max(int(training_cfg.get("batch_size", 16)), 1)
    grad_accum_steps = max(int(training_cfg.get("grad_accum_steps", 1)), 1)
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    max_seq_len = int(training_cfg.get("max_seq_len", 256))
    precision = str(training_cfg.get("precision", "fp32")).lower()
    weight_decay = float(training_cfg.get("weight_decay", 0.01))

    # ------------------------------------------------------------------
    # Load base + lora (fully frozen)
    # ------------------------------------------------------------------
    adapter_manager = BaseAdapterManager(config)
    adapter_manager.load_base_and_lora(strict=True)
    if adapter_manager.is_mock_base:
        raise RuntimeError("train_cls_head requires a real backbone, not mock.")

    base_model = adapter_manager.base_model
    assert base_model is not None

    for param in base_model.parameters():
        param.requires_grad_(False)

    hidden_dim = _get_hidden_dim(base_model)
    logger.info("hidden_dim=%d", hidden_dim)

    # ------------------------------------------------------------------
    # Build tokenizer
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    base_model_id = str(model_cfg.get("base_model", ""))
    tok_kw: dict = {"local_files_only": bool(model_cfg.get("local_files_only", False))}
    if model_cfg.get("trust_remote_code"):
        tok_kw["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, **tok_kw)

    # ------------------------------------------------------------------
    # Build classification head
    # ------------------------------------------------------------------
    head_device = adapter_manager.device
    cls_head = LinearClsHead(hidden_dim=hidden_dim).to(head_device)
    logger.info("LinearClsHead params: %d", sum(p.numel() for p in cls_head.parameters()))

    # ------------------------------------------------------------------
    # Load data (train split only)
    # ------------------------------------------------------------------
    if config.get("data", {}).get("real", {}).get("source_json"):
        manifest = load_real_dataset(config)
    else:
        manifest = generate_synthetic_dataset(config)
    train_dataset = AttackDataset.from_manifest(manifest, split="train")
    if len(train_dataset) == 0:
        raise RuntimeError("train split is empty")
    logger.info("Train samples: %d", len(train_dataset))

    # ------------------------------------------------------------------
    # Precompute static delta (lora_v4 fixed weights)
    # ------------------------------------------------------------------
    delta_weight = _static_delta(adapter_manager)

    # ------------------------------------------------------------------
    # Optimizer & precision context
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        cls_head.parameters(), lr=lr, weight_decay=weight_decay
    )
    use_amp = precision in {"bf16", "bfloat16", "fp16", "float16"} and head_device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    )

    input_device = adapter_manager.base_model_input_device()
    records = train_dataset.records

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    cls_head.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_accum = 0

        for i, record in enumerate(records):
            label = torch.tensor(
                [float(record["label"])], device=head_device
            )

            tokenized = _tokenize(record, tokenizer, max_seq_len, input_device)

            adapter_manager.inject_delta(delta_weight=delta_weight, mode="forward_hook")
            try:
                with amp_ctx:
                    with torch.no_grad():
                        hidden = last_hidden_state_from_base_model(base_model, **tokenized)

                    pooled = hidden.mean(dim=1).to(head_device)  # [1, H]
                    logit = cls_head(pooled).view(-1)            # [1]
                    loss = F.binary_cross_entropy_with_logits(logit, label)
                    loss_scaled = loss / grad_accum_steps
            finally:
                adapter_manager.clear_injection()

            loss_scaled.backward()
            epoch_loss += loss.item()
            n_accum += 1

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(records):
                torch.nn.utils.clip_grad_norm_(cls_head.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / max(n_accum, 1)
        logger.info("Epoch %d/%d | avg_loss=%.4f | steps=%d", epoch + 1, epochs, avg_loss, global_step)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(output)
    cls_head.save(output_path)
    logger.info("Saved cls_head to %s", output_path)

    return {"status": "ok", "output": str(output_path), "hidden_dim": hidden_dim, "epochs": epochs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen linear classification head for static_lora baseline.")
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for cls_head.pt. Defaults to eval.cls_head_path in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("dhr.train_cls_head")

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    output = args.output
    if output is None:
        output = config.get("eval", {}).get("cls_head_path")
    if not output:
        raise ValueError(
            "Specify --output or set eval.cls_head_path in config."
        )

    logger.info("Loaded config from %s", Path(args.config).resolve())
    logger.info("Output: %s", output)

    result = run_cls_head_training(config, output)
    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
