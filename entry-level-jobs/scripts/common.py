"""Shared helpers for the scripts in this folder."""
# %% Imports and project paths
# This module is imported by every script (directly or via llm.py), which is what makes the openpyxl
# patch below take effect project-wide. It has no third-party dependency beyond openpyxl itself.
from __future__ import annotations

import os
from pathlib import Path

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.worksheet.worksheet import Worksheet

# ROOT is the project folder (parent of scripts/). parents[1] assumes this file lives exactly one
# level below the project root; moving scripts/ deeper would silently point ROOT at the wrong place.
ROOT = Path(__file__).resolve().parents[1]
# The .env file is looked up at a fixed location rather than the working directory, so the scripts
# behave the same whether run from the project root or from elsewhere.
ENV_FILE = ROOT / ".env"


# %% Monkeypatch: strip Excel-illegal control characters on Worksheet.append
# Ad text and model output occasionally contain control characters (form feeds, nulls) that
# Excel cannot store; openpyxl raises IllegalCharacterError on append. Every script imports
# this module, so patching Worksheet.append here makes all review workbooks strip them.
#
# Background: one model rationale contained a form feed (\x0c) and crashed a whole review-workbook
# write at the very end of a paid labelling run. Rather than sanitising at each call site, the
# patch is applied once at import time. ILLEGAL_CHARACTERS_RE is openpyxl's own regex for the
# characters it refuses, so the patch stays in sync with whatever openpyxl considers illegal.
# Side effect to be aware of: this is a process-wide change to openpyxl behaviour; any other code
# in the same process that appends rows also gets its strings silently cleaned.
# The original bound method is kept so the patched version can delegate to it.
_orig_append = Worksheet.append


def _safe_append(self: Worksheet, iterable) -> None:
    # Only list/tuple rows are cleaned; openpyxl also accepts dicts and generators for append, and
    # those pass through untouched. The scripts only ever append lists, so this covers every case
    # they produce. Non-string cells (None, numbers) are left as-is.
    if isinstance(iterable, (list, tuple)):
        iterable = [ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v for v in iterable]
    return _orig_append(self, iterable)


# Install the patch. Happens at import, before any script builds a workbook.
Worksheet.append = _safe_append


# %% load_env: minimal .env reader with file-overrides-environment semantics
# Deliberate departure from python-dotenv's default: here the FILE wins over an existing shell
# variable. Reason: a stale DEEPSEEK_API_KEY exported in a shell session once shadowed the key in
# .env and caused confusing auth failures. If the usual behaviour (environment wins) is wanted,
# call load_env(override=False).
# Parsing is intentionally simple: no export prefix, no variable expansion, no multi-line values.
# Quotes around the value are stripped but quotes inside the value are kept.
def load_env(path: Path = ENV_FILE, override: bool = True) -> None:
    """Load KEY=VALUE lines from .env into os.environ.

    By default values in .env override existing environment variables, so a stale key left
    in the shell or inherited from a parent app cannot shadow the project's own config.
    Pass override=False to get the usual dotenv behaviour (environment wins).

    Format: one `KEY=VALUE` per line, `#` comments and blank lines ignored, optional quotes
    around the value. No dependency on python-dotenv.
    """
    # A missing .env is not an error: the key may legitimately come from the environment (CI, etc.).
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        # Skip blanks, full-line comments and anything without "="; trailing comments on a value
        # line are NOT stripped and would end up inside the value.
        if not line or line.startswith("#") or "=" not in line:
            continue
        # split on the first "=" only, so values containing "=" (e.g. base64) survive intact.
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value
