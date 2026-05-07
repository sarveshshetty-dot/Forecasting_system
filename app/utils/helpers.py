"""
helpers.py - Utility functions used across the system.
"""

import json
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def sanitize_state_name(state: str) -> str:
    """Convert state name to safe directory name."""
    return state.strip().replace(" ", "_").lower()


def state_model_dir(base_dir: Path, state: str, model_name: str) -> Path:
    return base_dir / sanitize_state_name(state) / model_name
