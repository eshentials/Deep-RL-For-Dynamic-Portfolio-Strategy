"""Batched Yahoo Finance downloads through yfinance's standard client.

Small batches with exponential backoff keep the request rate gentle. Returns wide
frames (Date x Ticker) for Close (split-adjusted), Adj Close (split and dividend
adjusted) and Volume. Nothing here cleans or winsorises: QA happens downstream and
every repair is logged.
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

FIELDS = ("Close", "Adj Close", "Volume")


def _one_batch(batch: list[str], start: str, end: str, retries: int) -> pd.DataFrame:
    for attempt in range(retries):
        df = yf.download(batch, start=start, end=end, auto_adjust=False, actions=False,
                         progress=False, threads=False, group_by="column",
                         multi_level_index=True)
        if df is not None and not df.empty:
            return df
        wait = 10 * 2 ** attempt
        print(f"  empty/limited response for {batch[:3]}...; retry in {wait}s")
        time.sleep(wait)
    return pd.DataFrame()


def download(tickers: list[str], start: str, end: str, chunk: int = 15,
             retries: int = 4, pause: float = 3.0) -> dict[str, pd.DataFrame]:
    """Download `tickers` over [start, end) (yfinance `end` is exclusive)."""
    out: dict[str, list[pd.DataFrame]] = {f: [] for f in FIELDS}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        df = _one_batch(batch, start, end, retries)
        if df.empty:
            print(f"  WARNING: no data for batch {batch}")
            continue
        for f in FIELDS:
            if f in df.columns.get_level_values(0):
                out[f].append(df[f].reindex(columns=batch))
        time.sleep(pause)
    frames = {}
    for f in FIELDS:
        if out[f]:
            wide = pd.concat(out[f], axis=1)
            wide.index = pd.to_datetime(wide.index).tz_localize(None).normalize()
            wide.index.name = "Date"
            frames[f] = wide.sort_index()
        else:
            frames[f] = pd.DataFrame()
    return frames
