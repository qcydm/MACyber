from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dhr.infer.adapt import run_adaptation
from dhr.utils.config import load_config
from dhr.utils.logging import setup_logger
from dhr.utils.seed import set_seed


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-sample adaptation entrypoint.")
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--input", default=None, help="Path to input json sample.")
    parser.add_argument("--checkpoint", default=None, help="Path to adaptation checkpoint.")
    parser.add_argument("--use_cache", default="true", help="true/false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("dhr.adapt")

    seed = int(config["project"]["seed"])
    set_seed(seed)

    logger.info("Loaded config from %s", Path(args.config).resolve())
    result = run_adaptation(
        config,
        input_path=args.input,
        checkpoint=args.checkpoint,
        use_cache=_parse_bool(args.use_cache),
    )
    logger.info("Adapt result: %s", result)


if __name__ == "__main__":
    main()
