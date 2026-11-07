"""Risk-free rate: RBI 91-day Treasury bill primary auction cut-off yields.

Source files: RBI exports (DBIE, or Handbook/WSS auction tables) placed in
data/raw/rbi/ (xlsx or csv). Every file there is parsed and the auction yields are
combined; overlapping auctions must agree. The parser finds the header row below
RBI's title rows and takes the implicit-yield column (never the cut-off price).

Each auction yield applies from its auction date to later sessions (forward fill,
past values only). Cash accrues (1 + y)^(calendar days / 365) - 1 between sessions.

Coverage is validated before anything is written: auctions must start on or before
REQUIRED_FROM (the earliest session any strategy or path feature reads) and reach
within MAX_STALE_DAYS of the data freeze. If no file exists, a PROVISIONAL constant
6.5% series and a marker file are written; protocol runs refuse to start while the
marker exists or while rf.parquet fails the same coverage check.

Run:  python -m data.rf               # build from data/raw/rbi/ (refuses if coverage is short)
      python -m data.rf --provisional # write the development placeholder
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from config import ROOT, load

RAW = ROOT / "data" / "raw" / "rbi"
PROC = ROOT / "data" / "processed"
MARKER = PROC / "rf_PROVISIONAL"
PROVISIONAL_RATE = 0.065
REQUIRED_FROM = pd.Timestamp("2013-11-01")  # first decisions are at the 2013 year-end close
MAX_STALE_DAYS = 14


def _is_yield_header(s: str) -> bool:
    s = s.lower()
    if "amount" in s:
        return False
    if "yield" in s:
        return True
    return ("cut-off" in s or "cut off" in s or "cutoff" in s) and "price" not in s


def _find_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Date column and 91-day cut-off yield column ("yield" beats "cut-off"; a price
    column is never a yield). A file holding only 91-day bills need not say '91'."""
    cols = [str(c) for c in df.columns]
    date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
    cands = [c for c in df.columns if _is_yield_header(str(c))]
    cands.sort(key=lambda c: ("yield" not in str(c).lower(), "91" not in str(c)))
    yld_col = cands[0] if cands else None
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


def _parse_dates(col: pd.Series) -> pd.DatetimeIndex:
    """Excel dates stay as they are; ISO strings (YYYY-MM-DD) parse as ISO; anything
    else (e.g. RBI's DD-MM-YYYY) parses day-first. Mixing rules per cell avoids pandas
    reading ISO '2020-01-08' as 1 August under dayfirst=True."""
    s = col.astype(object)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    is_dt = s.map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"))
    out[is_dt] = pd.to_datetime(s[is_dt], errors="coerce")
    txt = s[~is_dt].astype(str).str.strip()
    iso = txt.str.match(r"^\d{4}-\d{2}-\d{2}")
    out[txt.index[iso]] = pd.to_datetime(txt[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    out[txt.index[~iso]] = pd.to_datetime(txt[~iso], errors="coerce", dayfirst=True)
    return pd.DatetimeIndex(out)


def parse_rbi(path) -> pd.Series:
    df = _read_table(path)
    dcol, ycol = _find_columns(df)
    out = pd.Series(pd.to_numeric(df[ycol], errors="coerce").to_numpy(), index=_parse_dates(df[dcol]))
    out = out[out.index.notna()].dropna().sort_index()
    # Sub-header rows (column numbers, fiscal-year labels) must not become data:
    # keep plausible dates and yields only (percent between 0 and 30).
    out = out[(out.index >= pd.Timestamp("1990-01-01")) & (out > 0) & (out < 30)]
    out = out[~out.index.duplicated(keep="last")]
    if len(out) and out.median() > 1.0:  # percent -> decimal
        out = out / 100.0
    rng = f"{out.index.min().date()}..{out.index.max().date()}" if len(out) else "no rows"
    print(f"parsed {len(out)} auction yields from {getattr(path, 'name', path)}: "
          f"columns '{dcol}', '{ycol}', {rng}")
    return out


_PDF_ROW = re.compile(r"^(\d{2}-[A-Za-z]{3}-\d{4})\s+(\d{2}-[A-Za-z]{3}-\d{4})\s+(.+)$")


def parse_rbi_pdf(path) -> pd.Series:
    """RBI 'Auctions of 91-Day Government of India Treasury Bills' PDF (DBIE/WSS layout).

    Every auction row has 16 columns (missing values printed as '-'); column 13 is the
    implicit yield at cut-off price (percent, actual/365). Auctions with no accepted
    bids print '-' there and are skipped, so the previous yield carries forward.
    """
    from pypdf import PdfReader
    dates, vals, bad = [], [], 0
    for page in PdfReader(str(path)).pages:
        for line in (page.extract_text() or "").splitlines():
            m = _PDF_ROW.match(line.strip())
            if not m:
                continue
            toks = [m.group(1), m.group(2)] + m.group(3).split()
            if len(toks) != 16:
                bad += 1
                continue
            if toks[12] in ("-", ""):
                continue
            try:
                vals.append(float(toks[12].replace(",", "")))
                dates.append(pd.to_datetime(m.group(1), format="%d-%b-%Y"))
            except ValueError:
                bad += 1
    out = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    out = out[(out > 0) & (out < 30)]
    out = out[~out.index.duplicated(keep="last")] / 100.0
    rng = f"{out.index.min().date()}..{out.index.max().date()}" if len(out) else "no rows"
    print(f"parsed {len(out)} auction yields from {getattr(path, 'name', path)} (PDF; "
          f"{bad} malformed rows skipped), {rng}")
    return out


def load_yields(files) -> pd.Series:
    parts = [parse_rbi_pdf(f) if str(f).lower().endswith(".pdf") else parse_rbi(f) for f in files]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("no auction yields parsed from data/raw/rbi/")
    allv = pd.concat(parts)
    spread = allv.groupby(level=0).agg(lambda s: s.max() - s.min())
    bad = spread[spread > 1e-4]
    if len(bad):
        raise ValueError(f"files disagree on {len(bad)} auction dates, e.g. {bad.head(3).to_dict()}")
    return allv.groupby(level=0).first().sort_index()


def coverage_problems(yields: pd.Series, freeze: pd.Timestamp) -> list[str]:
    probs = []
    if yields.empty:
        return ["no auction yields"]
    if yields.index.min() > REQUIRED_FROM:
        probs.append(f"auctions start {yields.index.min().date()}, need on or before {REQUIRED_FROM.date()}")
    if (freeze - yields.index.max()).days > MAX_STALE_DAYS:
        probs.append(f"auctions end {yields.index.max().date()}, need within {MAX_STALE_DAYS} days of the "
                     f"freeze {freeze.date()}")
    gaps = yields.index.to_series().diff().dt.days
    if (gaps > 31).any():
        g = gaps[gaps > 31]
        probs.append(f"{len(g)} gaps over 31 days between auctions, first ending {g.index[0].date()}")
    return probs


def daily_accrual(yields: pd.Series, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """`yield` at session t = latest auction on or before t. The accrual for the
    interval (t-1, t] uses the yield known at the close of t-1, so an auction's yield
    only affects later sessions (protocol: 'from its result date to later sessions')."""
    y = yields.reindex(yields.index.union(sessions)).ffill().reindex(sessions).to_numpy()
    days = np.r_[1, np.diff(sessions.values).astype("timedelta64[D]").astype(int)]
    y_prev = np.r_[y[:1], y[:-1]]
    acc = (1.0 + y_prev) ** (days / 365.0) - 1.0
    return pd.DataFrame({"yield": y, "accrual": acc}, index=sessions)


def build(provisional: bool = False) -> pd.DataFrame:
    sessions = pd.read_parquet(PROC / "market.parquet").index
    freeze = pd.Timestamp(load()["data"]["freeze"])
    files = sorted(p for p in RAW.glob("*") if p.suffix.lower() in (".xls", ".xlsx", ".csv", ".pdf")) if RAW.exists() else []
    if provisional or not files:
        print("WARNING: writing a PROVISIONAL constant "
              f"{PROVISIONAL_RATE:.1%} series for development runs only.")
        out = daily_accrual(pd.Series(PROVISIONAL_RATE, index=sessions[:1]), sessions)
        out.to_parquet(PROC / "rf.parquet")
        MARKER.write_text("risk-free series is provisional; add the RBI 91-day T-bill export\n")
        return out
    yields = load_yields(files)
    yields = yields[yields.index >= pd.Timestamp("2012-01-01")]  # the study needs mid-2012 onward
    probs = coverage_problems(yields, freeze)
    if probs:
        raise SystemExit("refusing to build the risk-free series (outputs unchanged):\n  - " + "\n  - ".join(probs))
    out = daily_accrual(yields, sessions)
    if out.loc[out.index >= REQUIRED_FROM, "accrual"].isna().any():
        raise SystemExit("risk-free accrual has gaps after REQUIRED_FROM (outputs unchanged)")
    out.to_parquet(PROC / "rf.parquet")
    MARKER.unlink(missing_ok=True)
    return out


def is_provisional() -> bool:
    """True if the placeholder is in use OR the stored series fails the coverage check."""
    if MARKER.exists():
        return True
    p = PROC / "rf.parquet"
    if not p.exists():
        return True
    rf = pd.read_parquet(p)
    return bool(rf.loc[rf.index >= REQUIRED_FROM, "accrual"].isna().any())


if __name__ == "__main__":
    rf = build(provisional="--provisional" in sys.argv)
    print(rf.resample("YE")["yield"].mean().round(4).to_string())
