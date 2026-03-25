"""
dataset.py — Download & clean Indian equity + ETF + index data for Deep-RL portfolio.
Rewritten to use:
  - NSEPy for equities (official NSE historical)
  - NSE JSON API for NIFTY 50 index
  - NSE India ETF endpoint for NIFTYBEES
No Yahoo Finance. No rate limits.

Data-quality pipeline:
  Step 1 – Synchronise trading days
  Step 2 – Use Close as Adj Close equivalent
  Step 3 – Remove defective assets
  Step 4 – Outlier handling
"""

from __future__ import annotations

import os
import time
import random
import warnings
from datetime import date, datetime
import pandas as pd
import numpy as np
import requests
from nsepy import get_history

warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ───────────────────────────────────────────────────────────────────────────────

EQUITY_TICKERS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "KOTAKBANK",
]

ETF_TICKER = "NIFTYBEES"  # handled via NSE JSON API
INDEX_SYMBOL = "NIFTY 50"  # handled via NSEPy index=True

# ───────────────────────────────────────────────────────────────────────────────
# DATE WINDOWS
# ───────────────────────────────────────────────────────────────────────────────

FULL_START = date(2014, 1, 1)
FULL_END   = date(2024, 12, 31)

WINDOWS = {
    "train": (date(2014, 1, 1), date(2020, 12, 31)),
    "test":  (date(2021, 1, 1),  date(2021, 12, 31)),
    "val":   (date(2022, 1, 1),  date(2024, 12, 31)),
}

# ───────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ───────────────────────────────────────────────────────────────────────────────

MISSING_ROW_THRESHOLD   = 0.20
ZERO_VOL_MAX_DAYS       = 3
RETURN_WINSOR_THRESHOLD = 0.40
PRICE_JUMP_FACTOR       = 100
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# DOWNLOAD FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def download_equity_nsepy(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Download OHLCV for a single NSE equity using NSEPy."""
    try:
        df = get_history(
            symbol=symbol,
            start=start,
            end=end
        )
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"[ERROR] NSEPy failed for {symbol}: {e}")
        return pd.DataFrame()


def download_index_nifty50(start: date, end: date) -> pd.DataFrame:
    """Use NSEPy index=True to fetch NIFTY 50 index history."""
    try:
        df = get_history(
            symbol="NIFTY 50",
            index=True,
            start=start,
            end=end
        )
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"[ERROR] Index API failed: {e}")
        return pd.DataFrame()


def download_etf_niftybees(start: date, end: date) -> pd.DataFrame:
    """
    Scrape NIFTYBEES historical data from NSE's JSON API:
    https://www.nseindia.com/api/quote-historical?symbol=NIFTYBEES&series=[%22EQ%22]&from=...&to=...
    """
    try:
        s = start.strftime("%d-%m-%Y")
        e = end.strftime("%d-%m-%Y")
        url = (
            f"https://www.nseindia.com/api/quote-historical?"
            f"symbol=NIFTYBEES&series=[%22EQ%22]"
            f"&from={s}&to={e}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        sess = requests.Session()
        resp = sess.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        js = resp.json()

        rows = js.get("data", [])
        if not rows:
            print("[WARN] No ETF data found.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["historicalDate"], dayfirst=False)
        df = df.set_index("Date")

        df = df.rename(columns={
            "close": "Close",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "tradedVolume": "Volume",
        })

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"[ERROR] ETF scrape failed: {e}")
        return pd.DataFrame()

# ───────────────────────────────────────────────────────────────────────────────
# COMBINED DOWNLOAD
# ───────────────────────────────────────────────────────────────────────────────

def download_all_equities(start: date, end: date) -> pd.DataFrame:
    frames = {}

    for sym in EQUITY_TICKERS:
        print(f"Downloading equity: {sym}")
        df = download_equity_nsepy(sym, start, end)
        if df.empty:
            print(f"  WARNING: no data for {sym}.")
            continue

        df = df.rename(columns={"Close": "Adj Close"})  # treat Close as adjusted
        frames[sym] = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        time.sleep(random.uniform(0.5, 1.2))

    if not frames:
        raise RuntimeError("No equity data downloaded.")

    return pd.concat(frames, axis=1)


def download_index_and_etf(start: date, end: date) -> pd.DataFrame:
    print("Downloading NIFTY 50 index…")
    nifty = download_index_nifty50(start, end)

    print("Downloading NIFTYBEES ETF…")
    etf = download_etf_niftybees(start, end)

    out = {}
    if not nifty.empty:
        nifty = nifty.rename(columns={"Close": "Adj Close"})
        out["NIFTY50"] = nifty[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    if not etf.empty:
        etf = etf.rename(columns={"Close": "Adj Close"})
        out["NIFTYBEES"] = etf[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    if not out:
        return pd.DataFrame()

    return pd.concat(out, axis=1)

# ───────────────────────────────────────────────────────────────────────────────
# QUALITY PIPELINE
# ───────────────────────────────────────────────────────────────────────────────

def step1_sync_trading_days(adj_close: pd.DataFrame,
                             volume: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[Step 1] Sync trading days…")

    prices = adj_close.dropna(how="all")
    vol    = volume.dropna(how="all")

    common = prices.index.intersection(vol.index)
    prices = prices.loc[common]
    vol    = vol.loc[common]

    missing_frac = prices.isna().mean(axis=1)
    drop_dates = missing_frac[missing_frac > MISSING_ROW_THRESHOLD].index
    prices = prices.drop(index=drop_dates)
    vol    = vol.drop(index=drop_dates)

    vol = vol.ffill()

    print(f"  Remaining rows: {len(prices)}")
    return prices, vol


def step2_use_adj_close(equity_df: pd.DataFrame):
    print("\n[Step 2] Extracting Adj Close + Volume…")

    adj = equity_df.xs("Adj Close", axis=1, level=1, drop_level=False)
    adj.columns = adj.columns.droplevel(0)

    vol = equity_df.xs("Volume", axis=1, level=1, drop_level=False)
    vol.columns = vol.columns.droplevel(0)

    return adj, vol


def step3_remove_defective_assets(adj_close, volume):
    print("\n[Step 3] Removing defective assets…")

    drops = []

    for ticker in adj_close.columns:
        # Zero-volume streak
        is_zero = (volume[ticker].fillna(0) == 0)
        streak = 0
        max_streak = 0
        for v in is_zero:
            streak = streak + 1 if v else 0
            max_streak = max(max_streak, streak)
        if max_streak > ZERO_VOL_MAX_DAYS:
            print(f"  {ticker}: zero-volume streak {max_streak}")
            drops.append(ticker)
            continue

        # Continuous NaN price
        na_run = adj_close[ticker].isna()
        streak = 0
        max_na = 0
        for v in na_run:
            streak = streak + 1 if v else 0
            max_na = max(max_na, streak)
        if max_na > ZERO_VOL_MAX_DAYS:
            print(f"  {ticker}: continuous NaN {max_na}")
            drops.append(ticker)

    keep = [t for t in adj_close.columns if t not in drops]
    print(f"  Kept: {keep}")
    return adj_close[keep], volume[keep]


def step4_handle_outliers(adj_close):
    print("\n[Step 4] Outlier handling…")

    # Detect extreme jumps
    ratio = adj_close / adj_close.shift(1)
    bad = (ratio > PRICE_JUMP_FACTOR) | (ratio < 1/PRICE_JUMP_FACTOR)
    drop_dates = bad.any(axis=1)
    if drop_dates.sum():
        print(f"  Removing {drop_dates.sum()} dates with extreme jumps.")
        adj_close = adj_close.loc[~drop_dates]

    # Winsorise returns
    ret = adj_close.pct_change()
    mask = ret.abs() > RETURN_WINSOR_THRESHOLD
    n_ext = mask.sum().sum()
    if n_ext:
        print(f"  Winsorising {n_ext} return values.")
    ret_clipped = ret.clip(
        lower=-RETURN_WINSOR_THRESHOLD,
        upper= RETURN_WINSOR_THRESHOLD
    )

    first = adj_close.iloc[[0]]
    mult = (1 + ret_clipped.iloc[1:]).cumprod()

    return pd.concat([first, first.values * mult])

# ───────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ───────────────────────────────────────────────────────────────────────────────

def run_quality_pipeline(raw: pd.DataFrame):
    adj, vol = step2_use_adj_close(raw)
    adj, vol = step1_sync_trading_days(adj, vol)
    adj, vol = step3_remove_defective_assets(adj, vol)
    adj      = step4_handle_outliers(adj)
    vol      = vol.reindex(adj.index)
    return adj, vol


def split_and_save(df: pd.DataFrame, name: str):
    for split, (s, e) in WINDOWS.items():
        subset = df.loc[s:e]
        path = os.path.join(OUTPUT_DIR, f"{name}_{split}.csv")
        subset.to_csv(path)
        print(f"Saved {path} ({len(subset)} rows)")


# ───────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("STEP 1 — Downloading all equity data")
    print("="*60)
    eq_raw = download_all_equities(FULL_START, FULL_END)
    eq_raw.to_csv(os.path.join(OUTPUT_DIR, "equity_raw_full.csv"))

    print("\nSTEP 2 — Downloading NIFTY50 index + NIFTYBEES ETF")
    idx_etf_raw = download_index_and_etf(FULL_START, FULL_END)
    idx_etf_raw.to_csv(os.path.join(OUTPUT_DIR, "index_etf_raw_full.csv"))

    print("\nSTEP 3 — Running data-quality pipeline")
    adj_clean, vol_clean = run_quality_pipeline(eq_raw)

    adj_clean.to_csv(os.path.join(OUTPUT_DIR, "adj_close_clean_full.csv"))
    vol_clean.to_csv(os.path.join(OUTPUT_DIR, "volume_clean_full.csv"))

    split_and_save(adj_clean, "adj_close_clean")
    split_and_save(vol_clean, "volume_clean")

    if not idx_etf_raw.empty:
        split_and_save(idx_etf_raw.xs("Adj Close", level=0, axis=1), "index_etf")

    print("\nAll files written to ./data/")