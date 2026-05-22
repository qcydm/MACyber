from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect per-sample eval traces (generated_text + parse status) "
            "from eval_summary.json."
        )
    )
    parser.add_argument("--summary", required=True, help="Path to eval_summary.json")
    parser.add_argument("--baseline", default=None, help="Filter baseline")
    parser.add_argument("--split", default=None, help="Filter split")
    parser.add_argument("--status", default=None, help="Filter parse_status")
    parser.add_argument("--limit", type=int, default=20, help="Number of sample rows to print")
    parser.add_argument(
        "--max_chars",
        type=int,
        default=320,
        help="Preview length for generated_text when --full_text=false",
    )
    parser.add_argument("--full_text", default="false", help="true/false")
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid summary JSON structure: {path}")
    return payload


def _trace_files_from_summary(summary: dict[str, Any], summary_path: Path) -> list[Path]:
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    trace_paths: list[Path] = []
    trace_map = artifacts.get("sample_trace_paths")
    if isinstance(trace_map, dict):
        for raw in trace_map.values():
            p = Path(str(raw)).resolve()
            if p.exists():
                trace_paths.append(p)

    if trace_paths:
        return sorted(set(trace_paths))

    trace_dir_raw = artifacts.get("sample_trace_dir")
    if trace_dir_raw is None:
        trace_dir = (summary_path.parent / "sample_traces").resolve()
    else:
        trace_dir = Path(str(trace_dir_raw)).resolve()

    if not trace_dir.exists():
        return []
    return sorted(trace_dir.glob("*.jsonl"))


def _shorten(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")

    summary = _load_summary(summary_path)
    trace_files = _trace_files_from_summary(summary, summary_path)
    if not trace_files:
        print(f"No trace files found from summary: {summary_path}")
        print("Hint: run evaluation again with the updated evaluator to generate sample_traces/*.jsonl")
        return

    selected_baseline = args.baseline.strip() if isinstance(args.baseline, str) and args.baseline else None
    selected_split = args.split.strip() if isinstance(args.split, str) and args.split else None
    selected_status = args.status.strip() if isinstance(args.status, str) and args.status else None
    show_full_text = _parse_bool(args.full_text)

    status_counter = Counter()
    per_group_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    filtered: list[dict[str, Any]] = []

    for trace_file in trace_files:
        with trace_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue

                baseline = str(row.get("baseline", ""))
                split = str(row.get("split", ""))
                parse_status = str(row.get("parse_status", ""))

                if selected_baseline and baseline != selected_baseline:
                    continue
                if selected_split and split != selected_split:
                    continue
                if selected_status and parse_status != selected_status:
                    continue

                status_counter[parse_status] += 1
                per_group_counter[(baseline, split)][parse_status] += 1

                row["_trace_file"] = str(trace_file)
                row["_line_no"] = line_no
                filtered.append(row)

    if not filtered:
        print("No records matched the current filters.")
        return

    print(f"Summary: {summary_path}")
    print(f"Matched records: {len(filtered)}")
    print("Parse status counts:")
    for status, count in status_counter.most_common():
        print(f"  - {status or '<empty>'}: {count}")

    print("Per baseline/split:")
    for (baseline, split), counter in sorted(per_group_counter.items()):
        total = sum(counter.values())
        parts = ", ".join(f"{k or '<empty>'}={v}" for k, v in counter.most_common())
        print(f"  - {baseline} / {split}: total={total}; {parts}")

    limit = max(int(args.limit), 0)
    if limit == 0:
        return

    print("\nSample outputs:")
    for idx, row in enumerate(filtered[:limit], start=1):
        baseline = str(row.get("baseline", ""))
        split = str(row.get("split", ""))
        sample_id = str(row.get("sample_id", ""))
        parse_status = str(row.get("parse_status", ""))
        pred_action = str(row.get("pred_action", ""))
        pred_severity = str(row.get("pred_severity", ""))
        pred_official = str(row.get("pred_official", ""))
        generated_text = str(row.get("generated_text", ""))

        if not show_full_text:
            generated_text = _shorten(generated_text.replace("\n", "\\n"), args.max_chars)

        print(
            f"[{idx}] baseline={baseline} split={split} sample_id={sample_id} "
            f"status={parse_status} pred=({pred_action}, {pred_severity}, {pred_official})"
        )
        print(f"     generated_text: {generated_text}")


if __name__ == "__main__":
    main()
