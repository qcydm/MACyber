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
    if not rows:
        return "No rows."

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


def _parse_ok_ratio(trace_path: Path) -> float | None:
    if not trace_path.exists():
        return None
    total = 0
    ok = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        total += 1
        if bool(payload.get("parse_ok", False)):
            ok += 1
    if total == 0:
        return None
    return ok / total


def _extract_eval_rows(label: str, eval_summary_path: Path) -> list[dict[str, Any]]:
    summary = _load_json(eval_summary_path)
    rows = summary.get("rows", [])
    artifacts = summary.get("artifacts", {})
    sample_trace_paths = artifacts.get("sample_trace_paths", {}) if isinstance(artifacts, dict) else {}

    extracted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        baseline = row.get("baseline", "")
        split = row.get("split", "")
        trace_key = f"{baseline}:{split}"
        parse_ok_ratio = None
        trace_path_raw = sample_trace_paths.get(trace_key)
        if trace_path_raw:
            parse_ok_ratio = _parse_ok_ratio(Path(trace_path_raw))

        extracted.append(
            {
                "run": label,
                "baseline": baseline,
                "split": split,
                "sample_count": row.get("sample_count"),
                "exact_match": row.get("exact_match"),
                "action_accuracy": row.get("action_accuracy", row.get("accuracy")),
                "severity_accuracy": row.get("severity_accuracy"),
                "official_accuracy": row.get("official_accuracy"),
                "parse_ok_ratio": parse_ok_ratio,
                "adapt_latency_avg_ms": row.get("adapt_latency_avg_ms"),
                "effective_rank_mean": row.get("effective_rank_mean"),
                "ood_gap": row.get("ood_gap"),
            }
        )
    return extracted


def _extract_train_row(label: str, train_metrics_path: Path | None) -> dict[str, Any] | None:
    if train_metrics_path is None or not train_metrics_path.exists():
        return None
    rows = _load_jsonl(train_metrics_path)
    if not rows:
        return None
    latest = rows[-1]
    return {
        "run": label,
        "epoch": latest.get("epoch"),
        "train_loss": latest.get("train_loss"),
        "val_loss": latest.get("val_loss"),
        "train_effective_rank": latest.get("train_effective_rank"),
        "val_effective_rank": latest.get("val_effective_rank"),
    }


def compare_runs(run_specs: list[tuple[str, Path, Path | None]]) -> str:
    eval_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for label, eval_summary_path, train_metrics_path in run_specs:
        eval_rows.extend(_extract_eval_rows(label, eval_summary_path))
        train_row = _extract_train_row(label, train_metrics_path)
        if train_row is not None:
            train_rows.append(train_row)

    parts = [
        "评测对比",
        _render_table(
            eval_rows,
            [
                "run",
                "baseline",
                "split",
                "sample_count",
                "exact_match",
                "action_accuracy",
                "severity_accuracy",
                "official_accuracy",
                "parse_ok_ratio",
                "adapt_latency_avg_ms",
                "effective_rank_mean",
                "ood_gap",
            ],
        ),
    ]

    if train_rows:
        parts.extend(
            [
                "",
                "训练对比",
                _render_table(
                    train_rows,
                    [
                        "run",
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_effective_rank",
                        "val_effective_rank",
                    ],
                ),
            ]
        )

    return "\n".join(parts)


def _parse_run_specs(args_runs: list[list[str]]) -> list[tuple[str, Path, Path | None]]:
    parsed: list[tuple[str, Path, Path | None]] = []
    for item in args_runs:
        if len(item) not in {2, 3}:
            raise ValueError("--run expects: LABEL EVAL_SUMMARY [TRAIN_METRICS]")
        label = item[0]
        eval_summary = Path(item[1]).resolve()
        train_metrics = Path(item[2]).resolve() if len(item) == 3 else None
        if not eval_summary.exists():
            raise FileNotFoundError(f"eval summary not found: {eval_summary}")
        if train_metrics is not None and not train_metrics.exists():
            raise FileNotFoundError(f"train metrics not found: {train_metrics}")
        parsed.append((label, eval_summary, train_metrics))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple feature-pack experiment outputs side-by-side"
    )
    parser.add_argument(
        "--run",
        nargs="+",
        action="append",
        required=True,
        metavar="RUN_ARG",
        help="One run spec: LABEL EVAL_SUMMARY [TRAIN_METRICS]",
    )
    args = parser.parse_args()

    run_specs = _parse_run_specs(args.run)
    print(compare_runs(run_specs))


if __name__ == "__main__":
    main()
