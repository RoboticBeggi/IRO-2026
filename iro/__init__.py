"""IRO 2026 robot software: shared config, hardware helpers and the doctor."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config" / "robot.yaml") as f:
        return yaml.safe_load(f)
