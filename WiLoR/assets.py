"""Location of the runtime model assets.

Assets are read from ``weights/`` next to this package unless the environment
variable ``HANDPOSE_WEIGHTS`` points to another directory with the same layout.
"""
import os
from pathlib import Path


def weights_dir() -> Path:
    override = os.environ.get("HANDPOSE_WEIGHTS")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "weights"
