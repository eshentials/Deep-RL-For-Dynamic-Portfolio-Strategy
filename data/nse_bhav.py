"""NSE end-of-day bhavcopy: the exchange's official daily prices, used to check flagged moves.

A day whose vendor (Yahoo) return exceeds the jump threshold is accepted as a genuine move only
when (a) NSE's own close and previous close give the same return, so the vendor series matches
what traded, and (b) a reviewed explanation names the event and no corporate action falls on
that day. Note that NSE's previous close is NOT adjusted for a demerger: a special pre-open
session sets the reference price, which the bhavcopy shows as the day's open. Demergers are
therefore repaired (data/prices.py DEMERGERS), never "confirmed" by this check.

Files (plain downloads; the archive has no bot wall):
  to 2024-07-05:  /content/historical/EQUITIES/{YYYY}/{MON}/cm{DD}{MON}{YYYY}bhav.csv.zip
  from 2024-07-08: /content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip   (UDiFF format)
Zips are cached under scratch/ (ignored); the rows used are committed, with each file's URL and
SHA-256, in <study>/data/raw/nse_bhavcopy/rows.csv.
"""
from __future__ import annotations

import hashlib
import io
import time
import zipfile

import pandas as pd
import requests

from config import ROOT, study_root

HOST = "https://nsearchives.nseindia.com"
UA = {"User-Agent": "research-data-builder/1.0 (point-in-time index membership study)"}
CACHE = ROOT / "scratch" / "nse_bhavcopy"
ROWS = study_root() / "data" / "raw" / "nse_bhavcopy" / "rows.csv"
UDIFF_FROM = pd.Timestamp("2024-07-08")
COLS = ["date", "isin", "symbol", "series", "open", "close", "prevclose", "url", "zip_sha256"]


def url_for(day: pd.Timestamp, udiff: bool | None = None) -> str:
    day = pd.Timestamp(day)
    if udiff is None:
        udiff = day >= UDIFF_FROM
    if udiff:
        return f"{HOST}/content/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    mon = day.strftime("%b").upper()
    return f"{HOST}/content/historical/EQUITIES/{day:%Y}/{mon}/cm{day:%d}{mon}{day:%Y}bhav.csv.zip"


def _download(url: str) -> bytes | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / url.rsplit("/", 1)[-1]
    if path.exists():
        return path.read_bytes()
    for k in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and r.content[:2] == b"PK":
                path.write_bytes(r.content)
                time.sleep(1.5)
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(10 * (k + 1))
    return None


def _parse(raw: bytes, url: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip() for c in df.columns]
    if "TckrSymb" in df.columns:   # UDiFF format
        out = pd.DataFrame({"date": pd.to_datetime(df["TradDt"]).dt.date, "isin": df["ISIN"],
                            "symbol": df["TckrSymb"], "series": df["SctySrs"], "open": df["OpnPric"],
                            "close": df["ClsPric"], "prevclose": df["PrvsClsgPric"]})
    else:
        out = pd.DataFrame({"date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y").dt.date,
                            "isin": df["ISIN"], "symbol": df["SYMBOL"], "series": df["SERIES"],
                            "open": df["OPEN"], "close": df["CLOSE"], "prevclose": df["PREVCLOSE"]})
    out["url"] = url
    out["zip_sha256"] = hashlib.sha256(raw).hexdigest()
    return out[out["series"].astype(str).str.strip() == "EQ"]


def row(day, isins: set[str], symbols: set[str]) -> dict | None:
    """The EQ row for one security on one session (matched by ISIN, else by any of its symbols).
    Rows already looked up are read from the committed table, so builds are reproducible offline."""
    day = pd.Timestamp(day).date()
    have = pd.read_csv(ROWS, dtype=str) if ROWS.exists() else pd.DataFrame(columns=COLS)
    hit = have[(have["date"] == str(day)) & (have["isin"].isin(isins) | have["symbol"].isin(symbols))]
    if len(hit):
        r = hit.iloc[0].to_dict()
        return {k: (float(r[k]) if k in ("open", "close", "prevclose") else r[k]) for k in COLS}
    for udiff in (None, not (pd.Timestamp(day) >= UDIFF_FROM)):
        url = url_for(day, udiff)
        raw = _download(url)
        if raw is None:
            continue
        df = _parse(raw, url)
        m = df[df["isin"].isin(isins) | df["symbol"].isin(symbols)]
        if len(m):
            r = m.iloc[0].to_dict()
            r["date"] = str(r["date"])
            ROWS.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([have, pd.DataFrame([r])[COLS]]).to_csv(ROWS, index=False)
            return r
    return None
