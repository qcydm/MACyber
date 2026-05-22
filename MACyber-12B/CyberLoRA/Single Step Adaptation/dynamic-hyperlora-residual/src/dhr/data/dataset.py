from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from dhr.data.schemas import GenerativeSample


class AttackDataset:
    def __init__(self, records: Iterable[Mapping[str, Any] | GenerativeSample]) -> None:
        self.records = [self._normalize_record(record) for record in records]

    @staticmethod
    def _normalize_record(record: Mapping[str, Any] | GenerativeSample) -> dict[str, Any]:
        if isinstance(record, GenerativeSample):
            return record.to_dict()
        return GenerativeSample.from_mapping(record).to_dict()

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "AttackDataset":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"AttackDataset jsonl not found: {file_path}")

        records: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Invalid JSON object at {file_path}:{line_no}")
                records.append(payload)
        return cls(records=records)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], split: str) -> "AttackDataset":
        files = manifest.get("files", {})
        if not isinstance(files, Mapping) or split not in files:
            raise KeyError(f"split={split!r} not found in manifest files")
        return cls.from_jsonl(files[split])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]
