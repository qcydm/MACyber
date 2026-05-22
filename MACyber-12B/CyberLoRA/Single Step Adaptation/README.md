# Dynamic HyperLoRA-Residual

Experimental repository for **Dynamic HyperLoRA–Residual**, including:

- Core PoC code (training, single-step adaptation, evaluation)
- Sample tests
- Design notes and references

## Repository layout

```text
.
├── dynamic-hyperlora-residual/   # Main project code (runs end-to-end)
├── data/                         # Data files
├── references/                   # Papers and references
├── description.md                # Approach overview
└── our_plan.md                   # Development plan
```

## Quick start

The main scripts are under `dynamic-hyperlora-residual/`:

```bash
cd dynamic-hyperlora-residual
conda env create -f environment.yml
conda run -n dhr python train_meta.py --config configs/exp/poc_qwen06b.yaml
```
