"""Protocol loader. Every module reads parameters through `load()`."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "config" / "protocol.yaml"


@lru_cache(maxsize=1)
def load() -> dict:
    with open(PROTOCOL) as f:
        return yaml.safe_load(f)


def protocol_hash() -> str:
    """SHA-256 of the protocol file bytes (quoted in protocol.md and the ledger)."""
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
