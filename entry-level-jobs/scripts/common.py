"""Shared helpers for the scripts in this folder."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE, override: bool = True) -> None:
    """Load KEY=VALUE lines from .env into os.environ.

    By default values in .env override existing environment variables, so a stale key left
    in the shell or inherited from a parent app cannot shadow the project's own config.
    Pass override=False to get the usual dotenv behaviour (environment wins).

    Format: one `KEY=VALUE` per line, `#` comments and blank lines ignored, optional quotes
    around the value. No dependency on python-dotenv.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value
