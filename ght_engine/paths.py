"""Project paths shared by the engine and dataset loaders."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(
    os.environ.get("GRAPHHT_DATASET_DIR") or ROOT_DIR / "datasets"
).expanduser().resolve()
