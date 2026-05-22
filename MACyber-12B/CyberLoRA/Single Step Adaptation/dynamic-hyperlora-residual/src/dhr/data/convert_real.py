from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from dhr.data.schemas import GenerativeSample, label_from_output

# OOD dataset file stem → attack_family registry; update when adding new datasets
_OOD_STEM_TO_FAMILY: dict[str, str] = {
    "ICS": "ics_control",
    "Log_unknown": "ssh_log",
    "host": "host_behavior",
    "PhiUSIIL_URL_Dataset": "url_phishing",
    "SDN-DDoS_Traffic_Dataset": "sdn_ddos",
}


def _load_alpaca_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file that contains a list of alpaca-format records."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list, got {type(raw).__name__}: {path}")
    return raw


def _build_sample(
    item: dict[str, Any],
    sample_id: str,
    attack_family: str,
    split: str,
) -> GenerativeSample:
    """Build a GenerativeSample from a single alpaca-format record."""
    instruction = str(item.get("instruction", "")).strip()
    inp = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()
    if not instruction:
        raise ValueError(f"Record {sample_id} has empty 'instruction'")
    if not output:
        raise ValueError(f"Record {sample_id} has empty 'output'")
    label = label_from_output(output)
    return GenerativeSample(
        sample_id=sample_id,
        attack_family=attack_family,
        split=split,
        instruction=instruction,
        input=inp,
        output=output,
        label=label,
    )


def _load_ood_samples(ood_source_jsons: list[str]) -> list[GenerativeSample]:
    """Load OOD samples from multiple alpaca-format JSON files."""
    samples: list[GenerativeSample] = []
    global_idx = 0
    for path_str in ood_source_jsons:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"OOD source not found: {p}")
        family = _OOD_STEM_TO_FAMILY.get(p.stem, p.stem.lower())
        raw = _load_alpaca_json(p)
        for item in raw:
            samples.append(
                _build_sample(
                    item=item,
                    sample_id=f"ood-{global_idx:05d}",
                    attack_family=family,
                    split="test_ood",
                )
            )
            global_idx += 1
    return samples


def _write_jsonl(path: Path, records: list[GenerativeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in records:
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")


def load_real_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Convert alpaca-format JSON to GenerativeSample JSONL splits.

    Expects each record to have 'instruction', 'input', 'output' fields.
    The 0/1 label is derived from the JSON block embedded in 'output'.

    Returns a manifest dict compatible with the rest of the pipeline.
    """
    real_cfg = config.get("data", {}).get("real", {})
    source_json = real_cfg.get("source_json", "")
    if not source_json:
        raise ValueError("data.real.source_json must be set to use real data loader")

    source_path = Path(source_json)
    if not source_path.exists():
        raise FileNotFoundError(f"Real data source not found: {source_path}")

    output_dir = Path(str(real_cfg.get("output_dir", "data/real")))
    seed = int(real_cfg.get("seed", config.get("project", {}).get("seed", 42)))
    train_ratio = float(real_cfg.get("train_ratio", 0.74))
    val_ratio = float(real_cfg.get("val_ratio", 0.13))

    rng = random.Random(seed)
    raw = _load_alpaca_json(source_path)

    samples: list[GenerativeSample] = []
    for idx, item in enumerate(raw):
        samples.append(
            _build_sample(
                item=item,
                sample_id=f"real-{idx:05d}",
                attack_family="network_traffic",
                split="train",  # placeholder, overwritten below
            )
        )

    rng.shuffle(samples)
    total = len(samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test_id = total - n_train - n_val

    ood_source_jsons: list[str] = real_cfg.get("ood_source_jsons", [])
    ood_samples = _load_ood_samples(ood_source_jsons) if ood_source_jsons else []

    split_slices: dict[str, list[GenerativeSample]] = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test_id": samples[n_train + n_val :],
        "test_ood": ood_samples,
    }

    for split_name, split_samples in split_slices.items():
        for s in split_samples:
            s.split = split_name

    files: dict[str, str] = {}
    actual_counts: dict[str, int] = {}
    label_stats: dict[str, dict[str, int]] = {}
    family_stats: dict[str, dict[str, int]] = {}

    for split_name, split_samples in split_slices.items():
        split_path = output_dir / f"{split_name}.jsonl"
        _write_jsonl(split_path, split_samples)
        files[split_name] = str(split_path)
        actual_counts[split_name] = len(split_samples)
        label_stats[split_name] = dict(
            sorted(Counter(s.label for s in split_samples).items())
        )
        family_stats[split_name] = dict(
            sorted(Counter(s.attack_family for s in split_samples).items())
        )

    label_dist = Counter(s.label for s in samples)
    return {
        "seed": seed,
        "output_dir": str(output_dir),
        "counts": actual_counts,
        "files": files,
        "label_stats": label_stats,
        "family_stats": family_stats,
        "families": {
            "old": ["network_traffic"],
            "new": [],
            "ood": sorted({s.attack_family for s in ood_samples}),
        },
        "source": str(source_path),
        "total_label_dist": {str(k): int(v) for k, v in sorted(label_dist.items())},
        "split_sizes": {
            "n_train": n_train,
            "n_val": n_val,
            "n_test_id": n_test_id,
        },
    }
