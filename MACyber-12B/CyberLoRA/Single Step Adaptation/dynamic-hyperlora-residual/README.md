# Dynamic HyperLoRA-Residual (PoC v1)

This repository hosts a runnable PoC for `Dynamic HyperLoRA-Residual`.

Current milestone: `M5` (evaluation + baseline comparison + adaptation checkpoint loading).

## Quick start

```bash
conda env create -f environment.yml
conda run -n dhr python train_meta.py --config configs/exp/poc_qwen06b.yaml
conda run -n dhr python train_meta.py --config configs/exp/poc_qwen06b.yaml --dry_run true
conda run -n dhr python adapt_once.py --config configs/exp/poc_qwen06b.yaml --input sample.json --checkpoint checkpoints/poc_qwen06b/best.pt --use_cache true
conda run -n dhr python eval_ood.py --config configs/exp/poc_qwen06b.yaml --checkpoint checkpoints/poc_qwen06b/best.pt
conda run -n dhr python eval_ood.py --config configs/exp/poc_qwen06b.yaml --dry_run true
```

训练时会自动落盘每个 epoch 的指标到：

- `outputs/<experiment_name>/train_metrics/epoch_metrics.jsonl`
- `outputs/<experiment_name>/train_metrics/epoch_metrics.csv`
- `outputs/<experiment_name>/train_metrics/step_metrics.jsonl`（按优化 step 记录）
- `outputs/<experiment_name>/train_metrics/step_metrics.csv`
- `outputs/<experiment_name>/train_metrics/training_curves.html`（无额外依赖，浏览器直接打开）
- `outputs/<experiment_name>/train_metrics/training_curves.png`（若环境里安装了 `matplotlib`）

## Testing

Default smoke tests use `mock base + dry_run` and do not require loading a real LLM:

```bash
conda run -n dhr pytest -q tests/test_m4_training_smoke.py tests/test_m5_eval_smoke.py tests/test_cli_dry_run.py
```

Optional real-backbone smoke test (requires local model files and can be slow):

```bash
DHR_RUN_REAL_SMOKE=1 conda run -n dhr pytest -q -m real_backbone_smoke tests/test_real_backbone_smoke.py
```

## Multi-GPU

Use `model.device_map: auto` to shard the base model across visible GPUs.

Example config: `configs/exp/poc_qwen14b_multigpu.yaml`

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run -n dhr python train_meta.py --config configs/exp/poc_qwen14b_multigpu.yaml
```

Supported model keys:
- `device_map`: `auto` or an explicit map
- `max_memory`: per-device memory budget, e.g. `{cuda:0: 72GiB, cuda:1: 72GiB}`
- `base_model_dtype`: `bf16` / `fp16` / `fp32` / `auto`
- `low_cpu_mem_usage`: loader memory optimization flag

## Status

- [x] M1: skeleton, config loading, CLI entrypoints
- [x] M2: BaseAdapterManager, HyperResidualNet, MergeHyperNet, lora_ops (forward/injection dry run)
- [x] M3: FingerprintRetriever + synthetic dataset pipeline + train dry-run linkage
- [x] M4: training loop, validation, best/last checkpoint, rank pruning
- [x] M5: adaptation/evaluation full metrics and baseline comparison
