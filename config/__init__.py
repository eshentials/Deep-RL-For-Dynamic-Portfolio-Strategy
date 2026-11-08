"""Protocol loader. Every module reads parameters through `load()`.

The active protocol is config/protocol.yaml (protocol v1, Nifty 50) unless the
environment variable RB_PROTOCOL names another file in config/, for example

    RB_PROTOCOL=protocol_v2.yaml python -m data.market

Each protocol keeps its data, runs and results under its own study root (`study_root()`):
v1 at the repository root, as registered, and any other universe under studies/<universe>/.
A builder run for one study therefore cannot overwrite another study's files.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "config" / os.environ.get("RB_PROTOCOL", "protocol.yaml")


@lru_cache(maxsize=1)
def load() -> dict:
    with open(PROTOCOL) as f:
        return yaml.safe_load(f)


def protocol_hash() -> str:
    """SHA-256 of the protocol file bytes (quoted in protocol.md and the ledger)."""
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def protocol_tag() -> str:
    """The git tag that registers the active protocol."""
    return f"protocol-v{load()['version']}"


def study_root() -> Path:
    """Where the active protocol's data, runs and results live."""
    universe = load()["data"]["universe"]
    return ROOT if universe == "nifty50" else ROOT / "studies" / universe


def describe() -> str:
    cfg = load()
    return f"{PROTOCOL.relative_to(ROOT)} (v{cfg['version']}, {cfg['data']['universe']})"


def require_version(version: int, what: str) -> None:
    """Stop a study-specific script from running under another protocol."""
    if load()["version"] != version:
        raise SystemExit(f"{what} belongs to protocol v{version}; the active protocol is {describe()}")
