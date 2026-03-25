"""
dataset.py — Download & clean equity + macro data for Deep-RL portfolio.

Download architecture (three layers, tried in order):
  Layer 1 — Yahoo Finance v8 JSON API via a cookie-initialised session
             (no rate-limit, works reliably even on shared college WiFi)
  Layer 2 — yfinance Ticker.history()  (different Yahoo endpoint)
  Layer 3 — yfinance download() with threads=False, progress=False

Data-quality pipeline (run after every download):
  Step 1 — Synchronise trading days
             · Inner-join all assets on the date axis
             · Drop dates where > 20 % of assets are missing
             · Forward-fill Volume only — never prices, never returns
  Step 2 — Corporate-action adjustment
             · Use Adj Close throughout (splits / dividends / bonuses baked in)
             · Fall back to Close only when Adj Close is entirely missing
  Step 3 — Remove defective assets
             · Zero-volume streak > 3 consecutive days  → drop ticker
             · Continuous NaN-price window > 3 days     → drop ticker
  Step 4 — Outlier handling
             · |daily return| > 40 %   → winsorise to ± 40 %
             · Price ratio > 100 ×     → remove date (unadjusted-split artefact)
             · Price reconstructed from clipped returns (no forward leakage)
"""

from __future__ import annotations

import os
import time
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Universe ───────────────────────────────────────────────────────────────────
EQUITY_TICKERS: list[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "NIFTYBEES.NS",    # ETF — NIFTY 50 proxy
    "HINDUNILVR.NS",
    "KOTAKBANK.NS",
]

MACRO_TICKERS: dict[str, str] = {
    "INDIAVIX":  "^INDIAVIX",   # India VIX
    "CRUDE_OIL": "CL=F",        # Crude Oil Futures (USD)
    "USDINR":    "INR=X",       # USD / INR exchange rate
}

# ── Date windows ───────────────────────────────────────────────────────────────
FULL_START = "2014-01-01"
FULL_END   = "2024-12-31"

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2014-01-01", "2020-12-31"),
    "test":  ("2021-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2024-12-31"),
}

# ── Quality thresholds ─────────────────────────────────────────────────────────
MISSING_ROW_THRESHOLD   = 0.20   # drop date if > 20 % assets missing
ZERO_VOL_MAX_DAYS       = 3      # remove asset if zero-volume streak > this
NAN_PRICE_MAX_DAYS      = 3      # remove asset if NaN-price streak > this
RETURN_WINSOR           = 0.40   # winsorise |daily return| above this
PRICE_JUMP_FACTOR       = 100    # flag un-adjusted splits (100 × move)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD LAYERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared session (cookie initialised once, reused for every ticker) ──────────
_session: requests.Session | None = None

def _get_session() -> requests.Session:
    """Return a cookie-primed Yahoo Finance requests.Session (singleton)."""
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/html,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        # Prime cookies — required to avoid 401/rate-limit on subsequent calls
        try:
            s.get("https://finance.yahoo.com", timeout=10)
        except Exception:
            pass
        _session = s
    return _session


# ── Layer 1 — Raw Yahoo Finance v8 JSON API ────────────────────────────────────

def _yahoo_v8(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Call Yahoo Finance v8 chart API directly.
    Returns a flat OHLCV + Adj Close DataFrame indexed by Date.
    """
    try:
        t1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        t2 = int(datetime.strptime(end,   "%Y-%m-%d").timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&period1={t1}&period2={t2}&events=div%2Csplits"
        )
        resp = _get_session().get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        result = payload.get("chart", {}).get("result")
        if not result:
            return pd.DataFrame()

        r    = result[0]
        q    = r["indicators"]["quote"][0]
        ac   = r["indicators"].get("adjclose", [{}])[0]

        idx = pd.to_datetime(r["timestamp"], unit="s").normalize()
        df  = pd.DataFrame({
            "Open":      q.get("open"),
            "High":      q.get("high"),
            "Low":       q.get("low"),
            "Close":     q.get("close"),
            "Volume":    q.get("volume"),
            "Adj Close": ac.get("adjclose"),
        }, index=idx)
        df.index.name = "Date"

        # If Adj Close is missing entirely, use Close
        if df["Adj Close"].isna().all():
            df["Adj Close"] = df["Close"]

        return df.dropna(how="all")

    except Exception as exc:
        print(f"    [v8_api] {symbol}: {exc}")
        return pd.DataFrame()


# ── Layer 2 — yfinance Ticker.history() ───────────────────────────────────────

def _yf_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Use yfinance Ticker.history() — hits Yahoo's chart endpoint differently."""
    try:
        t  = yf.Ticker(symbol)
        df = t.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        return df
    except Exception as exc:
        print(f"    [yf_history] {symbol}: {exc}")
        return pd.DataFrame()


# ── Layer 3 — yfinance download() ─────────────────────────────────────────────

def _yf_download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: standard yfinance download with threading and progress disabled."""
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            threads=False,
            progress=False,
            timeout=20,
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        return df
    except Exception as exc:
        print(f"    [yf_download] {symbol}: {exc}")
        return pd.DataFrame()


# ── Orchestrator: cycle through layers with retry + back-off ──────────────────

def _fetch_one(symbol: str, start: str, end: str,
               retries: int = 3, base_wait: float = 8.0) -> pd.DataFrame:
    """
    Try v8_api → yf_history → yf_download.
    On total failure, back off exponentially and retry the whole cycle.
    """
    layers = [
        ("v8_api",     lambda: _yahoo_v8(symbol, start, end)),
        ("yf_history", lambda: _yf_history(symbol, start, end)),
        ("yf_download",lambda: _yf_download(symbol, start, end)),
    ]
    for attempt in range(retries):
        for name, fn in layers:
            df = fn()
            if not df.empty:
                print(f"    ✓  {symbol} via [{name}] — {len(df)} rows")
                return df
        wait = base_wait * (2 ** attempt) + random.uniform(1, 3)
        print(f"  All layers failed for {symbol}. "
              f"Waiting {wait:.1f}s (attempt {attempt + 1}/{retries}) …")
        time.sleep(wait)

    print(f"  ✗  Could not download {symbol} after {retries} attempts.")
    return pd.DataFrame()


# ── High-level download functions ─────────────────────────────────────────────

def download_equity(tickers: list[str],
                    start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV + Adj Close for every equity / ETF ticker.
    Returns a MultiIndex DataFrame (Field × Ticker).
    """
    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        print(f"  {ticker} …")
        df = _fetch_one(ticker, start, end)
        if df.empty:
            print(f"  ✗  {ticker} excluded — no data.")
            continue
        # Wrap flat columns into (Field, Ticker) MultiIndex
        mi = pd.MultiIndex.from_tuples(
            [(col, ticker) for col in df.columns], names=["Field", "Ticker"]
        )
        df.columns = mi
        per_ticker[ticker] = df
        time.sleep(random.uniform(1.0, 2.5))   # polite delay

    if not per_ticker:
        return pd.DataFrame()

    combined = pd.concat(per_ticker.values(), axis=1)
    combined.columns.names = ["Field", "Ticker"]
    return combined


def download_macro(ticker_map: dict[str, str],
                   start: str, end: str) -> pd.DataFrame:
    """Download Adj Close / Close for each macro series."""
    frames: dict[str, pd.Series] = {}
    for name, symbol in ticker_map.items():
        print(f"  {name} ({symbol}) …")
        df = _fetch_one(symbol, start, end)
        if df.empty:
            print(f"  ✗  {name} skipped — no data.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        frames[name] = df[col].rename(name)
        time.sleep(random.uniform(1.0, 2.5))

    return pd.concat(frames.values(), axis=1) if frames else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# DATA-QUALITY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def step2_extract_adj_close(equity_df: pd.DataFrame
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract Adj Close (corporate-action adjusted) and Volume.
    Falls back to Close when Adj Close is entirely absent.
    """
    print("\n[Step 2] Extracting Adj Close …")
    adj = equity_df["Adj Close"].copy()
    vol = equity_df["Volume"].copy()

    for col in adj.columns:
        if adj[col].isna().all():
            print(f"  WARNING: {col} Adj Close all-NaN — falling back to Close.")
            adj[col] = equity_df["Close"][col]

    print(f"  Tickers: {list(adj.columns)}")
    return adj, vol


def step1_sync_trading_days(adj: pd.DataFrame,
                             vol: pd.DataFrame
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inner-join dates, drop rows with > MISSING_ROW_THRESHOLD missing assets,
    forward-fill Volume only.
    """
    print(f"\n[Step 1] Synchronising trading days …")

    adj = adj.dropna(how="all")
    vol = vol.dropna(how="all")
    common = adj.index.intersection(vol.index)
    adj, vol = adj.loc[common], vol.loc[common]

    missing_frac = adj.isna().mean(axis=1)
    bad = missing_frac[missing_frac > MISSING_ROW_THRESHOLD].index
    if len(bad):
        print(f"  Dropping {len(bad)} date(s) with >{MISSING_ROW_THRESHOLD*100:.0f}% "
              f"assets missing.")
    adj = adj.drop(index=bad)
    vol = vol.drop(index=bad)

    # Forward-fill volume only — NEVER forward-fill prices or returns
    vol = vol.ffill()

    print(f"  {len(adj)} trading days retained.")
    return adj, vol


def step3_remove_defective_assets(adj: pd.DataFrame,
                                   vol: pd.DataFrame
                                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove tickers with zero-volume streaks or continuous NaN price windows."""
    print(f"\n[Step 3] Removing defective assets …")
    drop: list[str] = []

    for ticker in adj.columns:
        # Zero-volume streak
        is_zero = vol[ticker].fillna(0) == 0
        streak = max_streak = 0
        for v in is_zero:
            streak = streak + 1 if v else 0
            max_streak = max(max_streak, streak)
        if max_streak > ZERO_VOL_MAX_DAYS:
            print(f"  Removing {ticker}: zero-volume streak = {max_streak} d.")
            drop.append(ticker)
            continue

        # NaN-price streak
        is_nan = adj[ticker].isna()
        streak = max_nan = 0
        for v in is_nan:
            streak = streak + 1 if v else 0
            max_nan = max(max_nan, streak)
        if max_nan > NAN_PRICE_MAX_DAYS:
            print(f"  Removing {ticker}: continuous NaN-price streak = {max_nan} d.")
            drop.append(ticker)

    keep = [t for t in adj.columns if t not in drop]
    print(f"  Kept {len(keep)} assets: {keep}")
    return adj[keep], vol[keep]


def step4_handle_outliers(adj: pd.DataFrame) -> pd.DataFrame:
    """
    1. Remove dates with a > PRICE_JUMP_FACTOR × price change
       (unadjusted-split artefact).
    2. Winsorise |daily return| > RETURN_WINSOR.
    3. Reconstruct prices from winsorised returns (no leakage).
    """
    print(f"\n[Step 4] Outlier handling …")

    # Remove 100 × price-jump dates
    ratio = adj / adj.shift(1)
    bad   = (ratio > PRICE_JUMP_FACTOR) | (ratio < 1 / PRICE_JUMP_FACTOR)
    bad_dates = bad.any(axis=1)
    if bad_dates.sum():
        flagged = adj.index[bad_dates].tolist()
        print(f"  Removing {len(flagged)} date(s) with ≥{PRICE_JUMP_FACTOR}× "
              f"price jump.")
        adj = adj.loc[~bad_dates]

    # Winsorise extreme daily returns
    ret = adj.pct_change()
    n_extreme = int((ret.abs() > RETURN_WINSOR).sum().sum())
    if n_extreme:
        print(f"  Winsorising {n_extreme} return(s) with |ret| > "
              f"{RETURN_WINSOR * 100:.0f}%.")
    ret_clipped = ret.clip(lower=-RETURN_WINSOR, upper=RETURN_WINSOR)

    # Reconstruct price from clipped returns (keeps first row exact)
    first = adj.iloc[[0]]
    mult  = (1 + ret_clipped.iloc[1:]).cumprod()
    adj_clean = pd.concat([first, first.values * mult])
    return adj_clean


def run_quality_pipeline(equity_df: pd.DataFrame
                          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all four quality steps; return (clean_adj_close, clean_volume)."""
    adj, vol = step2_extract_adj_close(equity_df)
    adj, vol = step1_sync_trading_days(adj, vol)
    adj, vol = step3_remove_defective_assets(adj, vol)
    adj      = step4_handle_outliers(adj)
    vol      = vol.reindex(adj.index)
    return adj, vol


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def split_and_save(df: pd.DataFrame, name: str) -> None:
    for split, (s, e) in WINDOWS.items():
        subset = df.loc[s:e]
        path   = os.path.join(OUTPUT_DIR, f"{name}_{split}.csv")
        subset.to_csv(path)
        print(f"  Saved {path}  ({len(subset)} rows)")


def print_summary(adj: pd.DataFrame, vol: pd.DataFrame,
                  macro: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Universe : {list(adj.columns)}")
    print(f"  Rows     : {len(adj)}")
    print(f"  Range    : {adj.index[0].date()} → {adj.index[-1].date()}")
    print(f"  NaN prices : {adj.isna().sum().sum()}")
    print(f"  NaN volume : {vol.isna().sum().sum()}")
    if not macro.empty:
        print(f"  Macro    : {list(macro.columns)}")
    print()
    print("Adj Close (last 3 rows):")
    print(adj.tail(3).to_string())
    if not macro.empty:
        print("\nMacro (last 3 rows):")
        print(macro.tail(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── A. Download equities ───────────────────────────────────────────────────
    print("=" * 60)
    print("A — Downloading equity / ETF data")
    print("=" * 60)
    equity_raw = download_equity(EQUITY_TICKERS, FULL_START, FULL_END)
    if equity_raw.empty:
        raise RuntimeError("No equity data downloaded. Check internet connection.")
    equity_raw.to_csv(os.path.join(OUTPUT_DIR, "equity_raw_full.csv"))
    print(f"\n  Raw equity shape: {equity_raw.shape}")

    # ── B. Download macro ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("B — Downloading macro / VIX / FX data")
    print("=" * 60)
    macro_df = download_macro(MACRO_TICKERS, FULL_START, FULL_END)
    if not macro_df.empty:
        macro_df.to_csv(os.path.join(OUTPUT_DIR, "macro_full.csv"))
        split_and_save(macro_df, "macro")

    # ── C. Data-quality pipeline ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("C — Running data-quality pipeline")
    print("=" * 60)
    adj_clean, vol_clean = run_quality_pipeline(equity_raw)

    # ── D. Persist ─────────────────────────────────────────────────────────────
    adj_clean.to_csv(os.path.join(OUTPUT_DIR, "adj_close_clean_full.csv"))
    vol_clean.to_csv(os.path.join(OUTPUT_DIR, "volume_clean_full.csv"))

    print("\nSplitting clean Adj Close:")
    split_and_save(adj_clean, "adj_close_clean")
    print("\nSplitting clean Volume:")
    split_and_save(vol_clean, "volume_clean")

    print_summary(adj_clean, vol_clean, macro_df)
    print("\nAll files written to ./data/")
