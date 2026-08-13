"""Environment configuration.

Reads .env from the repository root without pulling in a dependency for it.
Real environment variables win over the file, so CI and one-off overrides work
the obvious way.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
NORMALIZED = DATA / "normalized"
STATE = DATA / "state"


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _dotenv().get(name) or default


def require(name: str) -> str:
    value = get(name)
    if not value:
        raise RuntimeError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


# --- named settings ---------------------------------------------------------

HYDRA_CLOUD_DATABASE = get("HYDRA_CLOUD_DATABASE", "glasshouse")
BULK_MODEL = get("OLLAMA_BULK_MODEL", "gpt-oss:120b-cloud")
ADJUDICATION_MODEL = get("OLLAMA_ADJUDICATION_MODEL", "gpt-oss:120b-cloud")
