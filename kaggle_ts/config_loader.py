from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple


try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "TOML config requires Python 3.11+ or the 'tomli' package"
        ) from exc


DEFAULT_FILENAMES: Tuple[str, ...] = ("config.toml", "kts.toml", "conf.toml")

_CONFIG_PATH: Path | None = None
_CONFIG_CACHE: Dict[str, Any] | None = None


def _resolve_config_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for name in DEFAULT_FILENAMES:
        candidate = Path(name)
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_config(explicit: str | None = None) -> tuple[Path | None, Dict[str, Any]]:
    path = _resolve_config_path(explicit)
    if path is None:
        return None, {}
    try:
        with path.open("rb") as f:
            data: Dict[str, Any] = tomllib.load(f)
    except FileNotFoundError:
        return None, {}
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc
    return path, data


def load_cli_config(explicit: str | None = None) -> tuple[Path | None, Dict[str, Any]]:
    """Return (path, data) for CLI defaults and cache globally."""

    global _CONFIG_CACHE, _CONFIG_PATH
    path, data = _load_config(explicit)
    _CONFIG_PATH = path
    _CONFIG_CACHE = data
    return path, data


def load_app_config(explicit: str | None = None) -> tuple[Path | None, Dict[str, Any]]:
    """Return (path, data) for general use, caching results.

    ``explicit`` overrides are resolved first. When omitted, the loader checks
    ``$KTS_CONFIG`` and finally falls back to ``DEFAULT_FILENAMES``.
    """

    global _CONFIG_CACHE, _CONFIG_PATH
    candidate = explicit or os.environ.get("KTS_CONFIG")
    if candidate:
        resolved = Path(candidate).expanduser().resolve()
        if _CONFIG_CACHE is not None and _CONFIG_PATH == resolved:
            return _CONFIG_PATH, _CONFIG_CACHE
        path, data = _load_config(str(resolved))
        _CONFIG_PATH = path if path is not None else resolved
        _CONFIG_CACHE = data
        return _CONFIG_PATH, _CONFIG_CACHE
    if _CONFIG_CACHE is not None and _CONFIG_PATH is not None:
        return _CONFIG_PATH, _CONFIG_CACHE
    path, data = _load_config(None)
    _CONFIG_PATH = path
    _CONFIG_CACHE = data
    return path, data


def get_config_section(name: str) -> Dict[str, Any]:
    """Case-insensitive lookup for a named table."""

    _, config = load_app_config()
    for key, value in config.items():
        if isinstance(value, dict) and key.lower() == name.lower():
            return value
    return {}


def get_config_value(section: str, key: str) -> Any:
    table = get_config_section(section)
    for candidate, value in table.items():
        if candidate.lower() == key.lower():
            return value
    return None


def split_cli_config(data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Split raw config into (global_defaults, per_command_defaults)."""

    if not data:
        return {}, {}
    globals_section = {
        k: v for k, v in data.items() if not isinstance(v, dict) or k == "defaults"
    }
    per_command: Dict[str, Dict[str, Any]] = {}
    # Explicit command sections take precedence over "defaults" shim
    for key, value in data.items():
        if isinstance(value, dict) and key != "defaults":
            per_command[key] = value
    # Allow a shared [defaults] table that applies to every subcommand
    defaults = data.get("defaults")
    if isinstance(defaults, dict):
        globals_section = {**defaults, **globals_section}
    # Remove command tables from globals
    for key in list(globals_section.keys()):
        if key in per_command:
            globals_section.pop(key)
    return globals_section, per_command
