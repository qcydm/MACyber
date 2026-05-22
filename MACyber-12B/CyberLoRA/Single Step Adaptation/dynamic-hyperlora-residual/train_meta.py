from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dhr.training.trainer import run_training
from dhr.utils.config import load_config
from dhr.utils.device import pick_device
from dhr.utils.io import ensure_dir
from dhr.utils.logging import setup_logger
from dhr.utils.seed import set_seed


def _parse_bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--dry_run", default="false", help="true/false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("dhr.train")

    seed = int(config["project"]["seed"])
    set_seed(seed)

    output_root = ensure_dir(config["project"]["output_root"])
    ckpt_root = ensure_dir(config["project"]["checkpoint_root"])
    device = pick_device()

    logger.info("Loaded config from %s", Path(args.config).resolve())
    logger.info("Device: %s | seed: %d", device, seed)
    logger.info("Output dir: %s | Checkpoint dir: %s", output_root, ckpt_root)

    result = run_training(config, dry_run=_parse_bool(args.dry_run))
    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
