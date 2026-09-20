"""Runtime configuration for CalorieSnap.

Settings resolve in this order: environment variable, then the JSON file
at CONFIG_PATH (default /data/caloriesnap_config.json), then the default.
The file lets keys change without recreating the container.
"""

import json
import os
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("CONFIG_PATH",
                             "/data/caloriesnap_config.json")
LOCAL_MODEL = "fastvlm-local"

# USD per million tokens as [input, output]. Override with MODEL_PRICES.
_DEFAULT_PRICES = {"claude-haiku-4-5": [1.0, 5.0], LOCAL_MODEL: [0.0, 0.0]}


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


def cloud_configured() -> bool:
    """True when the Foundry endpoint and key are both set."""
    return bool(cfg("FOUNDRY_ENDPOINT") and cfg("FOUNDRY_API_KEY"))


def models() -> List[str]:
    """Returns selectable models. The first entry is the default.

    Cloud deployments come from FOUNDRY_MODEL and FOUNDRY_MODELS. The
    air-gapped FastVLM path is offered when FASTVLM_URL is set, and
    becomes the default when no cloud key is configured.
    """
    default = cfg("FOUNDRY_MODEL", "claude-haiku-4-5")
    extra = [m.strip() for m in cfg("FOUNDRY_MODELS").split(",")]
    names = [default] + [m for m in extra if m and m != default]
    if cfg("FASTVLM_URL"):
        if cloud_configured():
            names.append(LOCAL_MODEL)
        else:
            names.insert(0, LOCAL_MODEL)
    return names


def prices() -> Dict[str, List[float]]:
    """Returns the per-model price table."""
    try:
        custom = json.loads(cfg("MODEL_PRICES", "{}"))
    except ValueError:
        custom = {}
    return {**_DEFAULT_PRICES, **custom}
