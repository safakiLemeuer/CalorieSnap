"""Settings for the WhatsApp bot. Environment first, then a JSON file."""

import json
import os

CONFIG_PATH = os.environ.get("WA_CONFIG_PATH", "/data/whatsapp_config.json")


def cfg(key: str, default: str = "") -> str:
    """Returns one setting as a string."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            return str(json.load(handle).get(key) or default)
    except (OSError, ValueError):
        return default


def cfg_int(key: str, default: int) -> int:
    """Returns one setting as an int, falling back on bad input."""
    try:
        return int(cfg(key, str(default)))
    except ValueError:
        return default
