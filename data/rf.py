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


def _is_yield_header(s: str) -> bool:
    s = s.lower()
    return ("yield" in s or "cut-off" in s or "cut off" in s or "cutoff" in s) and "amount" not in s


def _find_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Date column and 91-day cut-off yield column. A file that holds only 91-day bills
    (e.g. Handbook Table 218) need not repeat '91' in the column name."""
    cols = [str(c) for c in df.columns]
    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    yields = [c for c in df.columns if _is_yield_header(str(c))]
    with_91 = [c for c in yields if "91" in str(c)]
    yld_col = (with_91 or yields or [None])[0]
    if date_col is None or yld_col is None:
        raise ValueError(f"could not find date / 91-day yield columns in {cols}")
    return date_col, yld_col


def _read_table(path) -> pd.DataFrame:
    """Read an RBI export whose header row may sit below title rows."""
    excel = str(path).lower().endswith(("xls", "xlsx"))
    raw = pd.read_excel(path, header=None) if excel else pd.read_csv(path, header=None)
    for i in range(min(40, len(raw))):
        cells = [str(v) for v in raw.iloc[i].tolist()]
        if any("date" in c.lower() for c in cells) and any(_is_yield_header(c) for c in cells):
            df = raw.iloc[i + 1:].copy()
            df.columns = [c.strip() for c in cells]
            return df
    raise ValueError(f"no header row with a date and a yield column in the first 40 rows of {path}")


def parse_rbi(path) -> pd.Series:
    df = _read_table(path)
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
