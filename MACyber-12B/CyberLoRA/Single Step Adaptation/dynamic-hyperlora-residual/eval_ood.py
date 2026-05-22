from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dhr.utils.config import load_config
from dhr.utils.logging import setup_logger
from dhr.utils.seed import set_seed


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD evaluation entrypoint.")
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint.")
    parser.add_argument("--dry_run", default="false", help="true/false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("dhr.eval")

    seed = int(config["project"]["seed"])
    set_seed(seed)

    logger.info("Loaded config from %s", Path(args.config).resolve())
    if _parse_bool(args.dry_run):
        logger.info("Evaluation status: dry_run_ok")
        logger.info("Checkpoint arg: %s", args.checkpoint)
        return

    from dhr.eval.evaluator import run_evaluation

    result = run_evaluation(config, checkpoint=args.checkpoint)
    logger.info("Evaluation status: %s", result.get("status"))
    table_text = result.get("table_text")
    if isinstance(table_text, str) and table_text:
        logger.info("Evaluation table:\n%s", table_text)
    logger.info("Checkpoint used: %s", result.get("checkpoint_used"))
    artifacts = result.get("artifacts", {})
    logger.info("Artifacts: %s", artifacts)


if __name__ == "__main__":
    main()
