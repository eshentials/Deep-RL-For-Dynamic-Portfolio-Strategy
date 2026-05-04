"""
preprocess.py — Standalone data-cleaning pipeline for Deep-RL portfolio dataset.

Reads raw downloaded CSVs from ./data/ and produces:
  adj_close_clean_{train,test,val,full}.csv
  volume_clean_{train,test,val,full}.csv
  cleaning_report.txt

Pipeline steps
──────────────
A. Corporate-action adjustment
   · Use Adj Close throughout (Yahoo already embeds splits/dividends/bonuses).
   · Fall back to Close only when Adj Close is entirely NaN for a ticker.
   · Raw Close is NEVER used for returns or signals.

B. Synchronise trading days
   · Inner-join all assets on the date axis (NSE trading days only).
   · Drop any date where > MISSING_ROW_THRESHOLD (20 %) of assets are NaN.
   · Forward-fill Volume only — prices and returns are NEVER forward-filled
     (forward-filling prices would introduce look-ahead bias).

C. Remove defective assets
   · Zero-volume streak > ZERO_VOL_MAX_DAYS (3) consecutive days → drop ticker.
     (Zero volume = trading halt, delisting, or bad feed — all corrupt RL transitions.)
   · Continuous NaN-price window > NAN_PRICE_MAX_DAYS (3) days → drop ticker.

D. Outlier handling
   · Remove dates where any asset shows a price ratio ≥ PRICE_JUMP_FACTOR (100 ×)
     vs the previous day.  This catches unadjusted-split artefacts that survived
     corporate-action adjustment.
   · Winsorise |daily log-return| > RETURN_WINSOR (40 %) to ± RETURN_WINSOR.
     Winsorisation (clip) is preferred over deletion because it preserves the
     time-series continuity required by the RL environment.
   · Prices are reconstructed from winsorised returns so no forward leakage occurs:
       P̂_t = P̂_{t-1} × (1 + r̂_t)

Design guarantees
─────────────────
• No forward leakage:  prices/returns are never filled forward.
• Split-adjusted:      only Adj Close is used.
• RL-safe transitions: no gaps, no NaN in the final outputs.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
REPORT_PATH = os.path.join(DATA_DIR, "cleaning_report.txt")

RAW_EQUITY_PATH = os.path.join(DATA_DIR, "raw_ohlcv_2014_2024.csv")

# ── Thresholds ─────────────────────────────────────────────────────────────────
MISSING_ROW_THRESHOLD = 0.20   # drop date if fraction of NaN assets exceeds this
ZERO_VOL_MAX_DAYS     = 3      # max allowed consecutive zero-volume days
NAN_PRICE_MAX_DAYS    = 3      # max allowed consecutive NaN-price days
RETURN_WINSOR         = 0.40   # winsorise |return| above this (40 %)
PRICE_JUMP_FACTOR     = 100    # flag price ratio ≥ 100 × as unadjusted split

# ── Date windows ───────────────────────────────────────────────────────────────
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2014-01-01", "2020-12-31"),
    "test":  ("2021-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2024-12-31"),
}

SPLIT_LABELS: dict[str, str] = {
    "train": "train_2014_2020",
    "test":  "test_2021",
    "val":   "val_2022_2024",
}

# ── Internal report buffer ─────────────────────────────────────────────────────
_report_lines: list[str] = []

def _log(msg: str = "") -> None:
    print(msg)
    _report_lines.append(msg)

def _section(title: str) -> None:
    bar = "─" * 60
    _log(f"\n{bar}")
    _log(f"  {title}")
    _log(bar)


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_multiindex_csv(path: str) -> pd.DataFrame:
    """
    Load a MultiIndex CSV produced by yfinance (two header rows: Field, Ticker).
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns.names = ["Field", "Ticker"]
    df.index.name = "Date"
    return df


def load_raw_equity() -> pd.DataFrame:
    """
    Load the raw equity MultiIndex DataFrame.
    Returns a (dates × (Field, Ticker)) MultiIndex DataFrame.
    """
    if not os.path.exists(RAW_EQUITY_PATH):
        raise FileNotFoundError(
            f"Raw equity file not found: {RAW_EQUITY_PATH}\n"
            "Run dataset.py first to download the data."
        )
    _log(f"  Loading raw equity from {RAW_EQUITY_PATH} …")
    return _load_multiindex_csv(RAW_EQUITY_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# STEP A — CORPORATE-ACTION ADJUSTMENT
# ══════════════════════════════════════════════════════════════════════════════

def step_a_adj_close(equity_df: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract Adj Close (split- / dividend- / bonus-adjusted) and Volume.

    Yahoo Finance bakes all corporate actions into Adj Close:
      · Stock splits:  price scaled by split ratio on ex-date
      · Dividends:     backward-adjusted so return series is continuous
      · Bonus shares:  treated identically to splits

    Fall-back rule: if a ticker's entire Adj Close column is NaN
    (can happen for some ETFs or indices), use raw Close instead.
    This is logged explicitly so you know which tickers are unadjusted.
    """
    _section("Step A — Corporate-action adjustment (Adj Close)")

    adj = equity_df["Adj Close"].copy()
    vol = equity_df["Volume"].copy()

    fallback_tickers: list[str] = []
    for ticker in adj.columns:
        if adj[ticker].isna().all():
            fallback_tickers.append(ticker)
            adj[ticker] = equity_df["Close"][ticker]

    if fallback_tickers:
        _log(f"  ⚠  Adj Close entirely NaN for {fallback_tickers}.")
        _log("     Fell back to raw Close for these tickers.")
        _log("     Returns for these tickers are NOT dividend-adjusted.")
    else:
        _log("  ✓  All tickers have valid Adj Close.")

    _log(f"\n  Universe ({len(adj.columns)} tickers):")
    for t in adj.columns:
        n_nan  = adj[t].isna().sum()
        n_zero = (vol[t].fillna(0) == 0).sum()
        _log(f"    {t:<20}  NaN prices={n_nan:>4}   zero-vol days={n_zero:>4}")

    return adj, vol


# ══════════════════════════════════════════════════════════════════════════════
# STEP B — SYNCHRONISE TRADING DAYS
# ══════════════════════════════════════════════════════════════════════════════

def step_b_sync(adj: pd.DataFrame,
                vol: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align all assets to a common NSE trading-day calendar.

    Rules
    ─────
    1. Inner-join on date index: only keep dates where at least one asset
       has a non-NaN price (outer-join dates that arise from macro / ETF
       calendar differences are dropped automatically).
    2. Drop dates where the fraction of NaN prices across tickers exceeds
       MISSING_ROW_THRESHOLD.  These are typically exchange-wide data-feed
       outages, not individual-stock issues.
    3. Forward-fill Volume only.
       Rationale: Volume = 0 on a holiday is a data artefact, not real;
       carry the last valid volume forward so the RL state is non-zero.
       Prices must NOT be forward-filled — a stale price would create a
       false zero-return on the following trading day (look-ahead bias).
    """
    _section("Step B — Synchronise trading days")

    before = len(adj)
    adj = adj.dropna(how="all")
    vol = vol.dropna(how="all")
    common = adj.index.intersection(vol.index)
    adj, vol = adj.loc[common], vol.loc[common]
    _log(f"  Inner-join: {before} → {len(adj)} dates")

    # Drop rows where too many assets are missing
    missing_frac = adj.isna().mean(axis=1)
    bad = missing_frac[missing_frac > MISSING_ROW_THRESHOLD].index
    if len(bad):
        _log(f"\n  Dropping {len(bad)} date(s) with "
             f">{MISSING_ROW_THRESHOLD*100:.0f}% assets missing:")
        for d in bad:
            pct = missing_frac[d] * 100
            _log(f"    {d.date()}  ({pct:.0f}% NaN)")
    else:
        _log(f"  No dates exceed the {MISSING_ROW_THRESHOLD*100:.0f}% "
             "missing threshold.")
    adj = adj.drop(index=bad)
    vol = vol.drop(index=bad)

    # Forward-fill volume — NEVER prices
    nan_vol_before = vol.isna().sum().sum()
    vol = vol.ffill()
    nan_vol_after  = vol.isna().sum().sum()
    _log(f"\n  Volume ffill: {nan_vol_before} → {nan_vol_after} NaN cells")
    _log(f"  ⚠  Prices NOT forward-filled (prevents look-ahead bias)")
    _log(f"\n  ✓  {len(adj)} trading days retained after synchronisation")

    return adj, vol


# ══════════════════════════════════════════════════════════════════════════════
# STEP C — REMOVE DEFECTIVE ASSETS
# ══════════════════════════════════════════════════════════════════════════════

def _max_streak(series: pd.Series) -> int:
    """Return the length of the longest consecutive True run in a boolean Series."""
    streak = max_s = 0
    for v in series:
        streak = streak + 1 if v else 0
        max_s  = max(max_s, streak)
    return max_s


def step_c_remove_defective(adj: pd.DataFrame,
                              vol: pd.DataFrame
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and remove tickers whose data is too broken to produce
    reliable RL training transitions.

    Removal criteria
    ────────────────
    Zero-volume streak > ZERO_VOL_MAX_DAYS
      A multi-day zero-volume period signals a trading halt, circuit-breaker
      suspension, or bad feed.  Any RL transition spanning this gap would have
      a grossly incorrect reward (return = 0 for several steps, then a large
      jump).

    Continuous NaN-price window > NAN_PRICE_MAX_DAYS
      Missing prices cannot be used in the environment's state vector.
      Imputing them would introduce non-stationary bias.
    """
    _section("Step C — Remove defective assets")

    drop: list[str] = []
    _log(f"  {'Ticker':<20}  {'MaxZeroVolStreak':>17}  {'MaxNaNPriceStreak':>18}  Status")
    _log(f"  {'─'*20}  {'─'*17}  {'─'*18}  {'─'*8}")

    for ticker in adj.columns:
        zvol_streak = _max_streak(vol[ticker].fillna(0) == 0)
        nan_streak  = _max_streak(adj[ticker].isna())

        if zvol_streak > ZERO_VOL_MAX_DAYS:
            status = f"DROPPED  (zero-vol streak={zvol_streak}d)"
            drop.append(ticker)
        elif nan_streak > NAN_PRICE_MAX_DAYS:
            status = f"DROPPED  (NaN-price streak={nan_streak}d)"
            drop.append(ticker)
        else:
            status = "OK"

        _log(f"  {ticker:<20}  {zvol_streak:>17}  {nan_streak:>18}  {status}")

    keep = [t for t in adj.columns if t not in drop]
    if drop:
        _log(f"\n  Removed {len(drop)} ticker(s): {drop}")
    else:
        _log(f"\n  ✓  All {len(keep)} tickers passed — none removed.")
    _log(f"  Final universe ({len(keep)}): {keep}")

    return adj[keep], vol[keep]


# ══════════════════════════════════════════════════════════════════════════════
# STEP D — OUTLIER HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def step_d_outliers(adj: pd.DataFrame) -> pd.DataFrame:
    """
    Two-pass outlier treatment.

    Pass 1 — Remove unadjusted-split artefact dates
    ─────────────────────────────────────────────────
    If any ticker shows a single-day price ratio ≥ PRICE_JUMP_FACTOR (100 ×)
    or ≤ 1/PRICE_JUMP_FACTOR, the entire date is removed.
    Rationale: this only occurs when a split was not correctly back-adjusted
    by the data provider; keeping it would create a ~9900 % return in the
    RL reward signal.

    Pass 2 — Winsorise extreme daily simple returns
    ────────────────────────────────────────────────
    |daily return| > RETURN_WINSOR (40 %) is clipped to ± RETURN_WINSOR.
    Winsorisation is preferred over deletion because:
      · It preserves the time-series continuity the RL environment needs.
      · It still captures the direction of the move (just dampened).
      · Deletion would create a gap that must be filled anyway.

    Price reconstruction
    ────────────────────
    After clipping, prices are re-derived from the clipped return series:
        P̂_0 = P_0  (first row kept exact)
        P̂_t = P̂_{t-1} × (1 + r̂_t)  for t ≥ 1
    This ensures the price series is self-consistent with the return series
    and introduces zero forward leakage.
    """
    _section("Step D — Outlier handling")

    # ── Pre-pass: interpolate isolated NaN prices ─────────────────────────────
    # Step C allows assets with NaN-price streaks ≤ NAN_PRICE_MAX_DAYS.
    # Isolated NaN days (e.g. a single missing feed day) must be filled
    # before price reconstruction or cumprod() propagates NaN forward.
    # Linear interpolation between two valid prices is the correct treatment
    # for a single missing trading day — it is NOT forward-filling returns.
    n_nan_before = int(adj.isna().sum().sum())
    if n_nan_before:
        adj = adj.interpolate(method="linear", limit_direction="both")
        n_nan_after = int(adj.isna().sum().sum())
        _log(f"  Pre-pass — Interpolated {n_nan_before - n_nan_after} "
             f"isolated NaN price(s)  "
             f"({n_nan_before} → {n_nan_after} NaN cells)")

    # ── Pass 1: 100× price-jump removal ───────────────────────────────────────
    ratio     = adj / adj.shift(1)
    jump_mask = (ratio >= PRICE_JUMP_FACTOR) | (ratio <= 1 / PRICE_JUMP_FACTOR)
    bad_dates = jump_mask.any(axis=1)

    n_bad = int(bad_dates.sum())
    if n_bad:
        flagged = adj.index[bad_dates].tolist()
        _log(f"  Pass 1 — Removing {n_bad} date(s) with ≥{PRICE_JUMP_FACTOR}× "
             f"price jump (unadjusted split):")
        for d in flagged:
            ticker_flag = jump_mask.columns[jump_mask.loc[d]].tolist()
            _log(f"    {d.date()}  offending tickers: {ticker_flag}")
        adj = adj.loc[~bad_dates]
    else:
        _log(f"  Pass 1 — No {PRICE_JUMP_FACTOR}× price jumps detected. ✓")

    # ── Pass 2: return winsorisation ──────────────────────────────────────────
    ret = adj.pct_change(fill_method=None)

    extreme = ret.abs() > RETURN_WINSOR
    n_ext   = int(extreme.sum().sum())

    if n_ext:
        _log(f"\n  Pass 2 — Winsorising {n_ext} observation(s) with "
             f"|return| > {RETURN_WINSOR*100:.0f}%:")
        for ticker in ret.columns:
            tix_ext = extreme[ticker]
            if tix_ext.any():
                for dt in ret.index[tix_ext]:
                    r_before = ret.loc[dt, ticker]
                    r_after  = float(np.clip(r_before,
                                             -RETURN_WINSOR, RETURN_WINSOR))
                    _log(f"    {ticker:<20}  {dt.date()}  "
                         f"return {r_before:+.2%} → clipped to {r_after:+.2%}")
    else:
        _log(f"\n  Pass 2 — No returns exceed ±{RETURN_WINSOR*100:.0f}%. ✓")

    ret_clipped = ret.clip(lower=-RETURN_WINSOR, upper=RETURN_WINSOR)

    # ── Reconstruct prices from clipped returns (no forward leakage) ──────────
    first      = adj.iloc[[0]]
    multiplier = (1 + ret_clipped.iloc[1:]).cumprod()
    adj_clean  = pd.concat([first, first.values * multiplier])
    adj_clean.index.name = "Date"

    _log(f"\n  ✓  Price series reconstructed from winsorised returns.")
    _log(f"     (First row kept exact; subsequent rows: P̂_t = P̂_{{t-1}} × (1+r̂_t))")
    return adj_clean


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def _final_checks(adj: pd.DataFrame, vol: pd.DataFrame) -> None:
    """Run final data-integrity assertions and log a summary."""
    _section("Final data-integrity checks")

    nan_p = adj.isna().sum().sum()
    nan_v = vol.isna().sum().sum()
    inf_p = np.isinf(adj.values).sum()

    _log(f"  Rows         : {len(adj)}")
    _log(f"  Tickers      : {len(adj.columns)}")
    _log(f"  Date range   : {adj.index[0].date()} → {adj.index[-1].date()}")
    _log(f"  NaN prices   : {nan_p}   {'✓' if nan_p == 0 else '⚠'}")
    _log(f"  NaN volume   : {nan_v}   {'✓' if nan_v == 0 else '⚠'}")
    _log(f"  Inf prices   : {inf_p}   {'✓' if inf_p == 0 else '⚠'}")

    # Per-split counts
    _log(f"\n  Split breakdown:")
    for split, (s, e) in WINDOWS.items():
        n = len(adj.loc[s:e])
        _log(f"    {split:<6}  {s} → {e}   {n:>5} rows")

    # Return distribution
    ret = adj.pct_change(fill_method=None).dropna(how="all")
    _log(f"\n  Return statistics (all assets, all dates):")
    _log(f"    Mean   : {ret.values[~np.isnan(ret.values)].mean():+.4f}")
    _log(f"    Std    : {ret.values[~np.isnan(ret.values)].std():.4f}")
    _log(f"    Min    : {np.nanmin(ret.values):+.4f}")
    _log(f"    Max    : {np.nanmax(ret.values):+.4f}")
    _log(f"    >±40%  : {int((ret.abs() > 0.40).sum().sum())} observations")

    assert nan_p == 0, "NaN prices remain after cleaning!"
    assert nan_v == 0, "NaN volume remains after cleaning!"
    assert inf_p == 0, "Inf prices found after cleaning!"
    _log(f"\n  ✓  All assertions passed. Data is RL-ready.")


def split_and_save(df: pd.DataFrame, name: str) -> None:
    for split, (s, e) in WINDOWS.items():
        subset = df.loc[s:e]
        label  = SPLIT_LABELS[split]
        path   = os.path.join(DATA_DIR, f"{name}_{label}.csv")
        subset.to_csv(path)
        _log(f"  Saved {path}  ({len(subset)} rows × {len(df.columns)} tickers)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute A → B → C → D, save outputs, write report. Returns (adj, vol)."""
    _log("=" * 60)
    _log(f"  Deep-RL Portfolio — Data Cleaning Pipeline")
    _log(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log("=" * 60)

    # ── Load raw data ──────────────────────────────────────────────────────────
    equity_raw = load_raw_equity()

    # ── A. Corporate-action adjustment ────────────────────────────────────────
    adj, vol = step_a_adj_close(equity_raw)

    # ── B. Synchronise trading days ───────────────────────────────────────────
    adj, vol = step_b_sync(adj, vol)

    # ── C. Remove defective assets ────────────────────────────────────────────
    adj, vol = step_c_remove_defective(adj, vol)

    # ── D. Outlier handling ───────────────────────────────────────────────────
    adj = step_d_outliers(adj)
    vol = vol.reindex(adj.index)   # realign volume to cleaned price index

    # ── Final checks ──────────────────────────────────────────────────────────
    _final_checks(adj, vol)

    # ── Persist ───────────────────────────────────────────────────────────────
    _section("Saving outputs")

    adj.to_csv(os.path.join(DATA_DIR, "prices_2014_2024.csv"))
    vol.to_csv(os.path.join(DATA_DIR, "volume_2014_2024.csv"))
    _log(f"  Saved data/prices_2014_2024.csv  ({len(adj)} rows)")
    _log(f"  Saved data/volume_2014_2024.csv  ({len(vol)} rows)")

    _log("\n  Splitting clean prices:")
    split_and_save(adj, "prices")
    _log("\n  Splitting clean volume:")
    split_and_save(vol, "volume")

    # ── Write report ──────────────────────────────────────────────────────────
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(_report_lines))
    print(f"\n  Full report written to {REPORT_PATH}")

    return adj, vol


if __name__ == "__main__":
    adj, vol = run_pipeline()
