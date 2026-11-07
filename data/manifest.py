"""SHA-256 manifest of every raw and processed data file.

Run:  python -m data.manifest            # (re)write data/MANIFEST.json
      python -m data.manifest --verify   # recompute and compare; exit 1 on any mismatch
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

from config import ROOT, protocol_hash

MANIFEST = ROOT / "data" / "MANIFEST.json"
TRACKED_DIRS = ["data/raw", "data/processed", "data/roster"]
SOURCES = {
    "data/raw/roster_snapshots": "Wayback Machine captures of NSE/niftyindices Nifty 50 constituent files and NSE equity-stockIndices JSON (web.archive.org id_ endpoint)",
    "data/raw/market": "Yahoo Finance via yfinance (^NSEI, NIFTYBEES.NS, ^INDIAVIX, INR=X)",
    "data/raw/prices": "Yahoo Finance via yfinance (constituent adjusted closes)",
    "data/raw/rbi": "Reserve Bank of India 91-day T-bill auction cut-off yields: Handbook 2025-26 Table 218 (rbidocs.rbi.org.in, downloaded by hand) plus a DBIE export",
    "data/roster": "Derived: roster snapshots, dated change events, membership intervals",
    "data/processed": "Derived by this repository's builders",
}


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files():
    for d in TRACKED_DIRS:
        base = ROOT / d
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.name != ".DS_Store":
                    yield p


def _source(rel: str) -> str:
    best = max((k for k in SOURCES if rel.startswith(k)), key=len, default="")
    return SOURCES.get(best, "")


def build() -> dict:
    entries = {}
    for p in _files():
        rel = str(p.relative_to(ROOT))
        entries[rel] = {"sha256": _sha(p), "bytes": p.stat().st_size, "source": _source(rel)}
    doc = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "protocol_sha256": protocol_hash(), "files": entries}
    MANIFEST.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


def verify() -> bool:
    doc = json.loads(MANIFEST.read_text())
    ok = True
    for rel, meta in doc["files"].items():
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING  {rel}")
            ok = False
        elif _sha(p) != meta["sha256"]:
            print(f"CHANGED  {rel}")
            ok = False
    extra = {str(p.relative_to(ROOT)) for p in _files()} - set(doc["files"])
    for rel in sorted(extra):
        print(f"UNLISTED {rel}")
        ok = False
    print("manifest OK" if ok else "manifest MISMATCH")
    return ok


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    d = build()
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(d['files'])} files")
