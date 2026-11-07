"""Market series on the NSE session calendar.

- Sessions: every date with a ^NSEI close (NSE trading days), price_start..freeze.
- Benchmark: NIFTYBEES adjusted close (total return). Days where its return differs
  from the index return by more than `benchmark_repair_threshold` get the index
  return instead; every repair is written to data/processed/benchmark_repairs.csv.
- India VIX: NSE-hours series, used same-day.
- USD/INR: last Yahoo FX close on or before each session (the one-session lag is
  applied in env/features.py, because Yahoo's FX close falls after the NSE close).

Run:  python -m data.market
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import ROOT, load
from data import yahoo

RAW = ROOT / "data" / "raw" / "market"
PROC = ROOT / "data" / "processed"


def fetch(force: bool = False) -> dict[str, pd.DataFrame]:
    cfg = load()["data"]
    RAW.mkdir(parents=True, exist_ok=True)
    tickers = [cfg["index"], cfg["benchmark"], cfg["vix"], cfg["fx"]]
    paths = {f: RAW / f"yahoo_market_{f.replace(' ', '_').lower()}.csv" for f in yahoo.FIELDS}
    if not force and all(p.exists() for p in paths.values()):
        return {f: pd.read_csv(p, index_col="Date", parse_dates=True) for f, p in paths.items()}
    end = (date.fromisoformat(cfg["freeze"]) + timedelta(days=1)).isoformat()
    frames = yahoo.download(tickers, cfg["price_start"], end, chunk=len(tickers))
    for f, p in paths.items():
        frames[f].to_csv(p)
    return frames


def build() -> pd.DataFrame:
    cfg = load()["data"]
    frames = fetch()
    close, adj = frames["Close"], frames["Adj Close"]
    freeze = pd.Timestamp(cfg["freeze"])

    idx = close[cfg["index"]].dropna()
    idx = idx[(idx.index >= pd.Timestamp(cfg["price_start"])) & (idx.index <= freeze)]
    sessions = idx.index

    bench = adj[cfg["benchmark"]].reindex(sessions)
    vix = close[cfg["vix"]].reindex(sessions)
    fx_all = close[cfg["fx"]].dropna()
    fx = fx_all.reindex(sessions, method="ffill")  # last FX close on or before the session

    bench_tr, repairs, deviations = repair_benchmark(
        bench, idx, cfg["benchmark_glitch_threshold"], cfg["benchmark_deviation_log"])

    out = pd.DataFrame({
        "nsei": idx,
        "bench_tr": bench_tr,
        "bench_raw_adj": bench,
        "vix": vix.ffill(),          # ffill uses past values only
        "vix_missing": vix.isna(),
        "usdinr": fx,
    })
    out.index.name = "Date"
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / "market.parquet")
    repairs.to_csv(PROC / "benchmark_repairs.csv")
    deviations.to_csv(PROC / "benchmark_tracking_deviations.csv")
    return out


def repair_benchmark(level: pd.Series, index: pd.Series, glitch: float, log_gap: float):
    """Repair vendor glitches in the ETF level, keep genuine tracking deviations.

    A session is a glitch when the ETF/index level ratio sits more than `glitch` (log)
    away from the median ratio of the previous 10 valid sessions; a missing session is
    also repaired. Repaired levels are imputed as previous level x index growth, so the
    series stays anchored to the observed levels and nothing is double counted.
    Days whose ETF return differs from the index return by more than `log_gap` but are
    not glitches (ETF premium/discount in crashes) are kept and logged.
    """
    lv = level.copy()
    ratio = np.log(lv / index)
    ref = ratio.where(lv.notna()).rolling(10, min_periods=3).median().shift(1)
    is_glitch = (ratio - ref).abs() > glitch
    fix = lv.isna() | is_glitch
    fix.iloc[0] = False
    out = lv.copy()
    for i in np.flatnonzero(fix.to_numpy()):
        out.iloc[i] = out.iloc[i - 1] * index.iloc[i] / index.iloc[i - 1]
    repairs = pd.DataFrame({"raw_level": lv, "repaired_level": out,
                            "reason": np.where(lv.isna(), "missing", "level_glitch")})[fix]
    r_b, r_i = out.pct_change(fill_method=None), index.pct_change(fill_method=None)
    dev = (r_b - r_i).abs()
    deviations = pd.DataFrame({"r_bench": r_b, "r_index": r_i, "gap": dev})[(dev > log_gap) & ~fix]
    return out, repairs, deviations


def report(df: pd.DataFrame) -> None:
    rep = pd.read_csv(PROC / "benchmark_repairs.csv", index_col=0, parse_dates=True)
    devs = pd.read_csv(PROC / "benchmark_tracking_deviations.csv", index_col=0, parse_dates=True)
    print(f"sessions: {len(df)}  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"benchmark repairs: {len(rep)}")
    if len(rep):
        print(rep.round(4).to_string())
    print(f"tracking deviations kept (logged): {len(devs)}")
    if len(devs):
        print(devs.round(4).to_string())
    print(f"VIX missing sessions (filled forward): {int(df['vix_missing'].sum())}")
    print(f"USD/INR missing after alignment: {int(df['usdinr'].isna().sum())}")
    for y in (2019, 2020, 2025):
        s = df.loc[str(y), "bench_tr"]
        if len(s):
            dd = (s / s.cummax() - 1).min()
            print(f"{y}: benchmark return {s.iloc[-1] / s.iloc[0] - 1:+.2%}, max drawdown {dd:.2%}")
    tr = df["bench_tr"]
    yrs = (tr.index[-1] - tr.index[0]).days / 365.25
    print(f"benchmark CAGR {np.exp(np.log(tr.iloc[-1] / tr.iloc[0]) / yrs) - 1:.2%} over {yrs:.1f}y")


if __name__ == "__main__":
    report(build())
