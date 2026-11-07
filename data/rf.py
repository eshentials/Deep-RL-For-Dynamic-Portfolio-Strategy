"""Risk-free rate: RBI 91-day Treasury bill primary auction cut-off yields.

Source file: an export from RBI's Database on Indian Economy (DBIE) placed in
data/raw/rbi/ (xlsx or csv). The parser looks for a date column and a 91-day yield
column, so the exact export layout does not matter much; it prints what it found.

Each auction yield applies from its result date to later sessions (forward fill,
past values only). Cash accrues (1 + y)^(calendar days / 365) - 1 between sessions.

Until the RBI file exists, `build()` writes a PROVISIONAL constant 6.5% series and a
marker file. Provisional data is for development runs on 2014-2018 only; the test
runner and protocol-mode training refuse to start while the marker exists.

Run:  python -m data.rf
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROOT, load

RAW = ROOT / "data" / "raw" / "rbi"
PROC = ROOT / "data" / "processed"
MARKER = PROC / "rf_PROVISIONAL"
PROVISIONAL_RATE = 0.065


def _find_columns(df: pd.DataFrame) -> tuple[str, str]:
    date_col, yld_col = None, None
    for c in df.columns:
        s = str(c).lower()
        if date_col is None and "date" in s:
            date_col = c
        if yld_col is None and ("91" in s and ("yield" in s or "cut" in s or "rate" in s)):
            yld_col = c
    if date_col is None or yld_col is None:
        raise ValueError(f"could not find date / 91-day yield columns in {list(df.columns)}")
    return date_col, yld_col


def parse_rbi(path) -> pd.Series:
    df = pd.read_excel(path) if str(path).lower().endswith(("xls", "xlsx")) else pd.read_csv(path)
    dcol, ycol = _find_columns(df)
    out = pd.Series(pd.to_numeric(df[ycol], errors="coerce").to_numpy(),
                    index=pd.to_datetime(df[dcol], errors="coerce", dayfirst=True))
    out = out[out.index.notna()].dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.median() > 1.0:  # percent -> decimal
        out = out / 100.0
    print(f"parsed {len(out)} auction yields from {path.name}: columns '{dcol}', '{ycol}', "
          f"{out.index.min().date()}..{out.index.max().date()}, range {out.min():.2%}..{out.max():.2%}")
    return out


def daily_accrual(yields: pd.Series, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    y = yields.reindex(yields.index.union(sessions)).ffill().reindex(sessions)
    days = np.r_[1, np.diff(sessions.values).astype("timedelta64[D]").astype(int)]
    acc = (1.0 + y.to_numpy()) ** (days / 365.0) - 1.0
    return pd.DataFrame({"yield": y.to_numpy(), "accrual": acc}, index=sessions)


def build() -> pd.DataFrame:
    sessions = pd.read_parquet(PROC / "market.parquet").index
    files = sorted(p for p in RAW.glob("*") if p.suffix.lower() in (".xls", ".xlsx", ".csv")) if RAW.exists() else []
    if files:
        yields = parse_rbi(files[-1])
        if yields.index.min() > sessions[0]:
            raise ValueError(f"RBI series starts {yields.index.min().date()}, after the first session")
        MARKER.unlink(missing_ok=True)
    else:
        print("WARNING: no RBI file in data/raw/rbi/. Writing a PROVISIONAL constant "
              f"{PROVISIONAL_RATE:.1%} series for development runs only.")
        yields = pd.Series(PROVISIONAL_RATE, index=sessions[:1])
        MARKER.write_text("risk-free series is provisional; add the RBI 91-day T-bill export\n")
    out = daily_accrual(yields, sessions)
    out.to_parquet(PROC / "rf.parquet")
    return out


def is_provisional() -> bool:
    return MARKER.exists()


if __name__ == "__main__":
    rf = build()
    print(rf.resample("YE")["yield"].mean().round(4).to_string())
