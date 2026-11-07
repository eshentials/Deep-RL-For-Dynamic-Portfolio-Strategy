"""Point-in-time Nifty 50 roster from archived copies of NSE's own constituent files.

Step 1 (this module's `fetch`): pull every Wayback Machine capture of the official
constituent CSV (and of NSE's JSON index API), validate each one, and record when each
distinct membership set was first and last seen.

Step 2 (`events`): diff consecutive membership sets into add/remove events, each
bracketed by the last capture of the old set and the first capture of the new one.

Step 3 (`build`, in roster_build.py): date each event from NSE notices and expand into
membership intervals.

Run:  python -m data.roster fetch     # downloads + parses captures (cached)
      python -m data.roster events    # prints bracketed change events
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "roster_snapshots"
OUT_DIR = ROOT / "data" / "roster"

CDX = "https://web.archive.org/cdx/search/cdx"
UA = {"User-Agent": "research-data-builder/1.0 (point-in-time index membership study)"}
START = "20130101"  # nothing earlier is needed: the study starts 2014-01-01


@dataclass(frozen=True)
class Source:
    pattern: str            # CDX url pattern
    kind: str               # "csv" or "json"
    valid_until: str | None = None  # captures after this date are stale copies


SOURCES = [
    # The pre-rename file froze after CNX Nifty became NIFTY 50 (Nov 2015). Its content
    # on 2015-10-01 equals the new file's on 2016-03-10, and later captures repeat it,
    # so only captures up to 2016-03-10 are point-in-time evidence.
    Source("www.nseindia.com/content/indices/ind_niftylist.csv", "csv", "20160310"),
    Source("www.nseindia.com/content/indices/ind_nifty50list.csv", "csv"),
    Source("archives.nseindia.com/content/indices/ind_nifty50list.csv", "csv"),
    Source("nsearchives.nseindia.com/content/indices/ind_nifty50list.csv", "csv"),
    Source("www.niftyindices.com/IndexConstituent/ind_nifty50list.csv", "csv"),
    Source("www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050", "json"),
]


PAUSE = 5.0  # seconds between archive requests; the archive refuses bursts


def _get(url: str, params: dict | None = None, tries: int = 6) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=90)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # refused connection = rate limit
            last = repr(e)[:160]
        wait = 45 * (k + 1)
        print(f"  archive busy ({last}); waiting {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last})")


def cdx_captures(src: Source) -> pd.DataFrame:
    """All captures (including revisits) for one source, oldest first."""
    r = _get(CDX, params={"url": src.pattern, "output": "json",
                          "fl": "timestamp,original,statuscode,digest,mimetype",
                          "from": START})
    rows = r.json()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "original", "statuscode", "digest", "mimetype"])
    df = pd.DataFrame(rows[1:], columns=rows[0])
    if src.valid_until:
        df = df[df["timestamp"].str[:8] <= src.valid_until]
    df["source"] = src.pattern
    df["kind"] = src.kind
    return df.reset_index(drop=True)


def _slug(ts: str, original: str, kind: str) -> str:
    host = original.split("//", 1)[-1].split("/", 1)[0].replace(":80", "").replace(":443", "")
    return f"{ts}_{host}.{kind}"


def download(ts: str, original: str, kind: str) -> Path:
    """Raw capture bytes via the id_ endpoint (no Wayback rewriting), cached on disk."""
    path = RAW_DIR / _slug(ts, original, kind)
    if not path.exists():
        r = _get(f"https://web.archive.org/web/{ts}id_/{original}")
        path.write_bytes(r.content)
        time.sleep(PAUSE)  # be polite to the archive
    return path


def _norm(col: str) -> str:
    return "".join(ch for ch in col.lower() if ch.isalnum())


def parse_csv(raw: bytes) -> pd.DataFrame | None:
    """Official constituent CSV -> symbol/isin/company/industry, or None if not one."""
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    if "symbol" not in text[:300].lower():
        return None  # an HTML error page or redirect body
    df = pd.read_csv(io.StringIO(text))
    cols = {_norm(c): c for c in df.columns}
    need = {"symbol": "symbol", "isincode": "isin", "companyname": "company"}
    if not all(k in cols for k in need):
        return None
    out = pd.DataFrame({v: df[cols[k]].astype(str).str.strip() for k, v in need.items()})
    out["industry"] = df[cols["industry"]].astype(str).str.strip() if "industry" in cols else ""
    if "series" in cols:
        out = out[df[cols["series"]].astype(str).str.strip().isin(["EQ", "BE", "BZ", "nan"])]
    return out.reset_index(drop=True)


def json_payload_date(raw: bytes) -> pd.Timestamp | None:
    """The API's own timestamp. Several archive captures replay a different day's
    response (redirects), so the payload date, not the crawl date, dates the reading."""
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace"))
        return pd.Timestamp(pd.to_datetime(doc["timestamp"], format="%d-%b-%Y %H:%M:%S").date())
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def parse_json(raw: bytes) -> pd.DataFrame | None:
    """NSE equity-stockIndices JSON -> same columns (the index row itself is skipped)."""
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    if not isinstance(doc, dict) or "data" not in doc:
        return None
    rows = []
    for d in doc["data"]:
        sym = str(d.get("symbol", "")).strip()
        if not sym or sym.upper() == "NIFTY 50" or d.get("priority") == 1:
            continue
        meta = d.get("meta") or {}
        rows.append({"symbol": sym, "isin": str(meta.get("isin", "")).strip(),
                     "company": str(meta.get("companyName", "")).strip(),
                     "industry": str(meta.get("industry", "")).strip()})
    return pd.DataFrame(rows) if rows else None


def fetch() -> pd.DataFrame:
    """Download and parse every useful capture; write snapshot tables.

    Returns the snapshot index: one row per parsed capture with its member set hash.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cdx_cache = RAW_DIR / "cdx_captures.csv"
    if cdx_cache.exists():
        caps = pd.read_csv(cdx_cache, dtype=str)
    else:
        lists = []
        for s in SOURCES:
            lists.append(cdx_captures(s))
            time.sleep(PAUSE)
        caps = pd.concat(lists, ignore_index=True)
        caps.to_csv(cdx_cache, index=False)
    caps = caps.drop_duplicates(subset=["timestamp", "original"])

    members_rows, index_rows = [], []
    parsed_digests: dict[str, pd.DataFrame | None] = {}
    for _, c in caps.sort_values("timestamp").iterrows():
        ts, orig, kind, dg = c["timestamp"], c["original"], c["kind"], c["digest"]
        if kind == "csv":
            # One download per distinct content; revisits ("-") reuse the parsed digest.
            if dg not in parsed_digests:
                if c["statuscode"] != "200":
                    continue
                raw = download(ts, orig, kind).read_bytes()
                parsed_digests[dg] = parse_csv(raw)
            df = parsed_digests[dg]
            date = pd.Timestamp(ts[:8])
        else:
            if c["statuscode"] != "200":
                continue
            raw = download(ts, orig, kind).read_bytes()
            df = parse_json(raw)
            date = json_payload_date(raw)
        if df is None or df.empty or date is None:
            continue
        syms = sorted(df["symbol"])
        mhash = hashlib.sha256(",".join(syms).encode()).hexdigest()[:12]
        index_rows.append({"capture_ts": ts, "date": date, "crawl_date": pd.Timestamp(ts[:8]),
                           "source": c["source"], "kind": kind, "status": c["statuscode"],
                           "digest": dg, "n_members": len(syms), "members_hash": mhash})
        for _, m in df.iterrows():
            members_rows.append({"capture_ts": ts, "members_hash": mhash, **m.to_dict()})

    index = pd.DataFrame(index_rows).sort_values(["date", "capture_ts"]).reset_index(drop=True)
    members = pd.DataFrame(members_rows)
    index.to_csv(OUT_DIR / "snapshot_index.csv", index=False)
    members.to_csv(OUT_DIR / "snapshot_members.csv", index=False)
    return index


def membership_sets() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Distinct consecutive membership sets with first/last seen dates."""
    index = pd.read_csv(OUT_DIR / "snapshot_index.csv", parse_dates=["date"], dtype={"capture_ts": str})
    members = pd.read_csv(OUT_DIR / "snapshot_members.csv", dtype=str)
    sets = {h: set(g["symbol"]) for h, g in members.groupby("members_hash")}
    index = index.sort_values(["date", "capture_ts"])
    runs, prev = [], None
    for _, r in index.iterrows():
        h = r["members_hash"]
        if prev is None or h != prev["members_hash"]:
            prev = {"members_hash": h, "first_seen": r["date"], "last_seen": r["date"],
                    "n": r["n_members"], "captures": 1}
            runs.append(prev)
        else:
            prev["last_seen"] = r["date"]
            prev["captures"] += 1
    return pd.DataFrame(runs), sets


def isin_map() -> dict[str, str]:
    """symbol -> ISIN as printed in NSE's own files (first non-empty value seen)."""
    m = pd.read_csv(OUT_DIR / "snapshot_members.csv", dtype=str)
    m = m[m["isin"].notna() & (m["isin"].str.len() == 12) & (m["isin"].str[7:9] == "01")]
    return m.drop_duplicates("symbol").set_index("symbol")["isin"].to_dict()


def events() -> pd.DataFrame:
    """Bracketed changes; a removed and an added symbol with the same ISIN is a rename."""
    runs, sets = membership_sets()
    isin = isin_map()
    rows = []
    for a, b in zip(runs.itertuples(), runs.iloc[1:].itertuples()):
        old, new = sets[a.members_hash], sets[b.members_hash]
        added, removed = sorted(new - old), sorted(old - new)
        renames = [(r, s) for r in removed for s in added if isin.get(r) and isin.get(r) == isin.get(s)]
        for r, s in renames:
            added.remove(s)
            removed.remove(r)
        rows.append({"after": a.last_seen.date(), "before": b.first_seen.date(),
                     "added": " ".join(added), "removed": " ".join(removed),
                     "renamed": " ".join(f"{r}->{s}" for r, s in renames),
                     "n_old": len(old), "n_new": len(new)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    pd.set_option("display.width", 200, "display.max_colwidth", 80, "display.max_rows", 200)
    if cmd == "fetch":
        idx = fetch()
        print(idx[["capture_ts", "source", "kind", "n_members", "members_hash"]].to_string())
    elif cmd == "events":
        runs, _ = membership_sets()
        print(runs.to_string())
        print()
        print(events().to_string())
    else:
        raise SystemExit(f"unknown command {cmd}")
