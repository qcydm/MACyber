from __future__ import annotations

from typing import Any

from dhr.training.engine import TrainingEngine


def run_training(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    engine = TrainingEngine(config)
    if dry_run:
        summary = engine.dry_run()
        return {
            "mode": "dry_run",
            "status": summary.status,
            "steps": summary.steps,
            "loss": summary.loss,
            "m2_payload": summary.m2_payload,
        }
    return engine.run()

