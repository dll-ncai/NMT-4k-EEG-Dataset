"""Small configuration helpers.

The project deliberately uses one readable YAML file rather than a large
configuration framework. This keeps every experimental choice visible.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and attach the absolute project/config locations."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping of configuration sections.")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    """Resolve relative paths against the project root, not the current shell."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def save_config_snapshot(config: dict[str, Any], destination: Path) -> None:
    """Save exactly the settings used by a run, excluding private helper keys."""
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def require_sections(config: dict[str, Any], sections: list[str]) -> None:
    """Fail early when a required YAML section was accidentally removed."""
    missing = [name for name in sections if name not in config]
    if missing:
        raise KeyError(f"Missing required configuration sections: {missing}")

