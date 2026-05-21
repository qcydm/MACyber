#!/usr/bin/env python3
"""Validate MACyber benchmark JSON files."""
import argparse
import json
from pathlib import Path


SEVERITIES = {"benign", "suspicious", "low", "medium", "high"}
ACTIONS = {"none", "monitor", "block"}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_item(item, path, index, strict=False):
    errors = []
    warnings = []
    prefix = f"{path}:{index}"

    def add_issue(message):
        (errors if strict else warnings).append(message)

    if not isinstance(item, dict):
        message = f"{prefix}: item must be an object"
        return ([message], []) if strict else ([], [message])

    for key in ("meta", "json", "label", "reasoning", "response"):
        if key not in item:
            add_issue(f"{prefix}: missing '{key}'")

    meta = item.get("meta", {})
    if not isinstance(meta, dict):
        errors.append(f"{prefix}: meta must be an object")
    else:
        for key in ("category", "subcategory"):
            if not meta.get(key):
                target = errors if strict else warnings
                target.append(f"{prefix}: missing meta.{key}")

    label = item.get("label", {})
    if not isinstance(label, dict):
        errors.append(f"{prefix}: label must be an object")
    else:
        if not label.get("official"):
            add_issue(f"{prefix}: missing label.official")
        severity = label.get("severity")
        if severity not in SEVERITIES:
            add_issue(f"{prefix}: invalid label.severity={severity!r}")

    reasoning = item.get("reasoning", {})
    if not isinstance(reasoning, dict):
        add_issue(f"{prefix}: reasoning must be an object")
    else:
        evidence = reasoning.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            add_issue(f"{prefix}: reasoning.evidence must be a non-empty list")
        elif not all(isinstance(x, str) and x.strip() for x in evidence):
            add_issue(f"{prefix}: reasoning.evidence must contain non-empty strings")
        analysis = reasoning.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            add_issue(f"{prefix}: reasoning.analysis must be a non-empty string")

    response = item.get("response", {})
    if not isinstance(response, dict):
        errors.append(f"{prefix}: response must be an object")
    else:
        action = response.get("action")
        if action not in ACTIONS:
            target = errors if strict else warnings
            target.append(f"{prefix}: invalid response.action={action!r}")

    return errors, warnings


def iter_dataset_files(data_dir):
    for path in sorted(data_dir.glob("*/*.json")):
        if path.name == "taxonomy.json":
            continue
        yield path


def validate_taxonomy(data_dir, taxonomy_path):
    errors = []
    taxonomy = load_json(taxonomy_path)
    expected = {
        (category, dataset)
        for category, datasets in taxonomy.items()
        for dataset in datasets
    }
    actual = {
        (path.parent.name, path.stem)
        for path in iter_dataset_files(data_dir)
    }
    for item in sorted(expected - actual):
        errors.append(f"taxonomy: missing dataset file {item[0]}/{item[1]}.json")
    for item in sorted(actual - expected):
        errors.append(f"taxonomy: unexpected dataset file {item[0]}/{item[1]}.json")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate MACyber benchmark JSON schema.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[2] / "data"),
        help="MACyber data directory.",
    )
    parser.add_argument(
        "--taxonomy",
        default=None,
        help="Optional taxonomy JSON path. Defaults to <data-dir>/taxonomy.json if it exists.",
    )
    parser.add_argument("--max-errors", type=int, default=50, help="Stop after this many errors.")
    parser.add_argument("--strict", action="store_true", help="Treat all schema irregularities as errors.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    taxonomy_path = Path(args.taxonomy) if args.taxonomy else data_dir / "taxonomy.json"
    errors = []
    warnings = []
    total = 0
    files = 0

    if taxonomy_path.exists():
        errors.extend(validate_taxonomy(data_dir, taxonomy_path))

    for path in iter_dataset_files(data_dir):
        files += 1
        data = load_json(path)
        if not isinstance(data, list):
            errors.append(f"{path}: top-level JSON must be a list")
            continue
        for index, item in enumerate(data):
            total += 1
            item_errors, item_warnings = validate_item(item, path, index, strict=args.strict)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
            if len(errors) >= args.max_errors:
                break
        if len(errors) >= args.max_errors:
            break

    if errors:
        print(f"Validation failed: {len(errors)} error(s), {len(warnings)} warning(s), checked {total} records in {files} files.")
        for error in errors[: args.max_errors]:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Validation passed: checked {total} records in {files} files.")
    if warnings:
        print(f"Warnings: {len(warnings)} irregular record(s). Use --strict to treat them as errors.")
        for warning in warnings[: args.max_errors]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
