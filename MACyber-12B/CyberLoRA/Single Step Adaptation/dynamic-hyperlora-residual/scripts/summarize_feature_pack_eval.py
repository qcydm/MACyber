from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)


def _render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(_fmt(row.get(col))))

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    body = []
    for row in rows:
        body.append(" | ".join(_fmt(row.get(col)).ljust(widths[col]) for col in columns))
    return "\n".join([header, sep, *body])


def _pick_metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["baseline", "split"]
    keys = set(rows[0].keys())
    columns = ["baseline", "split", "sample_count"]
    generative = ["exact_match", "action_accuracy", "severity_accuracy", "official_accuracy"]
    binary = ["accuracy", "precision", "recall", "f1"]
    if "exact_match" in keys:
        columns.extend([c for c in generative if c in keys])
    else:
        columns.extend([c for c in binary if c in keys])
    for col in ["adapt_latency_avg_ms", "effective_rank_mean", "ood_gap"]:
        if col in keys:
            columns.append(col)
    return columns


def summarize(eval_summary_path: Path, train_metrics_path: Path | None) -> str:
    summary = _load_json(eval_summary_path)
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"rows must be a list in {eval_summary_path}")

    parts = [
        f"评测摘要：{eval_summary_path}",
        "",
        _render_table(rows, _pick_metric_columns(rows)),
    ]

    if train_metrics_path is not None and train_metrics_path.exists():
        epoch_rows = _load_jsonl(train_metrics_path)
        if epoch_rows:
            latest = epoch_rows[-1]
            parts.extend(
                [
                    "",
                    f"训练摘要：{train_metrics_path}",
                    json.dumps(
                        {
                            "epoch": latest.get("epoch"),
                            "train_loss": latest.get("train_loss"),
                            "val_loss": latest.get("val_loss"),
                            "train_effective_rank": latest.get("train_effective_rank"),
                            "val_effective_rank": latest.get("val_effective_rank"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval/train metrics for feature-pack experiments")
    parser.add_argument("--eval-summary", required=True, help="Path to outputs/eval/<exp>/eval_summary.json")
    parser.add_argument(
        "--train-metrics",
        help="Optional path to outputs/<exp>/train_metrics/epoch_metrics.jsonl",
    )
    args = parser.parse_args()

    eval_summary_path = Path(args.eval_summary).resolve()
    train_metrics_path = Path(args.train_metrics).resolve() if args.train_metrics else None

    if not eval_summary_path.exists():
        raise FileNotFoundError(f"eval summary not found: {eval_summary_path}")
    if train_metrics_path is not None and not train_metrics_path.exists():
        raise FileNotFoundError(f"train metrics not found: {train_metrics_path}")

    print(summarize(eval_summary_path, train_metrics_path))


if __name__ == "__main__":
    main()
