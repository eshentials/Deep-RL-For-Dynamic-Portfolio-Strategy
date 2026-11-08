"""Constituent prices from Yahoo, with coverage and jump QA over each membership window.

Yahoo keeps a renamed company's history under its current ticker, so candidates are
tried in order and the one with the best coverage over the symbol's membership
windows (plus the covariance warm-up before each) is used.

Classification (protocol Section 2): "adjusted" if coverage >= min_coverage and every
day with |return| > jump_flag is explained in KNOWN_EVENTS; otherwise "excluded" for
its membership months. Both lists are written to data/processed/price_qa.csv.

Run:  python -m data.prices
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import load, study_root
from data import yahoo

RAW = study_root() / "data" / "raw" / "prices"
PROC = study_root() / "data" / "processed"
ROSTER = study_root() / "data" / "roster"

CANDIDATES = {
    "MCDOWELL-N": ["UNITDSPR.NS", "MCDOWELL-N.NS"],
    "INFRATEL": ["INDUSTOWER.NS", "INFRATEL.NS"],
    "IBULHSGFIN": ["SAMMAANCAP.NS", "IBULHSGFIN.NS"],
    "TMPV": ["TMPV.NS", "TATAMOTORS.NS"],
    "ETERNAL": ["ETERNAL.NS", "ZOMATO.NS"],
    "LTIM": ["LTM.NS", "LTIM.NS"],
}

# Demergers Yahoo did not (fully) adjust. The true ex-date return is the ex-date close
# over NSE's special pre-open discovered price; prices before the ex-date are rescaled
# so the series shows exactly that return.
DEMERGERS = [
    # symbol, ex-date, discovered price, previous close (both NSE, rupees), source
    ("RELIANCE", "2023-07-20", 2580.00, 2841.85,
     "https://www.businesstoday.in/markets/company-stock/story/ril-share-price-settles-at-rs-2580-in-special-pre-open-session-jio-financial-at-rs-26185-390528-2023-07-20"),
    ("ITC", "2025-01-06", 455.60, 481.60,
     "https://es.tradingview.com/news/moneycontrol:e51894f6e094b:0-itc-hotels-demerger-itc-discovered-price-at-rs-455-6-down-rs-27-following-special-pre-open-session"),
    ("TMPV", "2025-10-14", 400.00, 660.75,
     "https://www.autocarpro.in/news/tata-motors-stock-adjusts-to-demerger-shares-of-pv-unit-start-trading-at-rs-400-129224"),
]

# Days with |return| > jump_flag inside a membership window that were reviewed by hand
# and are genuine price moves (no corporate action in the raw close).
_YES = "Yes Bank crisis 2018-2020 (RBI curtailed CEO tenure; asset-quality shocks; Mar-2020 moratorium and reconstruction)"
KNOWN_EVENTS: dict[tuple[str, str], str] = {
    ("ADANIENT", "2023-02-01"): "Hindenburg report sell-off; FPO withdrawn 2023-02-01",
    ("ADANIENT", "2023-02-02"): "Hindenburg report sell-off",
    ("AXISBANK", "2020-03-23"): "COVID crash: Nifty -13% on 2020-03-23",
    ("BAJAJFINSV", "2020-03-23"): "COVID crash: Nifty -13% on 2020-03-23",
    ("DLF", "2014-10-14"): "SEBI barred DLF from the capital market for three years (2014-10-13)",
    ("IDEA", "2017-01-30"): "Vodafone India-Idea merger talks confirmed",
    ("INDUSINDBK", "2020-03-26"): "Rebound from the COVID-crash low",
    ("INDUSINDBK", "2025-03-11"): "Disclosure of derivative-accounting discrepancy",
    ("JINDALSTEL", "2015-02-19"): "Rally on coal-block e-auction outcome (no corporate action in raw close)",
    ("SBIN", "2017-10-25"): "Government PSU-bank recapitalisation plan (2017-10-24)",
    ("TMPV", "2019-10-29"): "Two-session move including the 2019-10-27 Diwali muhurat session, which is not an index session",
    ("ZEEL", "2019-01-25"): "Promoter share-pledge sell-off",
}
for _d in ["2018-09-21", "2019-02-14", "2019-04-30", "2019-10-03", "2020-03-05", "2020-03-06",
           "2020-03-09", "2020-03-11", "2020-03-16", "2020-03-17"]:
    KNOWN_EVENTS[("YESBANK", _d)] = _YES


# Protocol v2 (Nifty Next 50) additions. Merged only when that protocol is active, so protocol
# v1's outputs (price_qa.csv, price_jumps_explained.csv, ...) cannot change.
CANDIDATES_NEXT50 = {
    "L&TFH": ["LTF.NS", "L&TFH.NS"],          # L&T Finance Holdings, now L&T Finance (same ISIN)
}
_BHAV_0430 = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20260430_F_0000.csv.zip"
DEMERGERS_NEXT50: list = [
    # Five-way demerger, ex-date 2026-04-30. NSE's bhavcopy shows the special pre-open discovered
    # price as the day's open (289.50) and the unadjusted previous close (773.60).
    ("VEDL", "2026-04-30", 289.50, 773.60,
     f"{_BHAV_0430} (open = special pre-open price); "
     "https://www.multibagg.ai/market-pulse/articles/vedanta-ex-demerger-price-fall-cmol1h3tirvh6pg0j4wog395i"),
]
# Reviewed by hand; under protocol v2 each is accepted only if NSE's bhavcopy return agrees.
_COVID = "COVID crash: Nifty -13% on 2020-03-23"
KNOWN_EVENTS_NEXT50: dict[tuple[str, str], str] = {
    ("BANKBARODA", "2017-10-25"): "Government PSU-bank recapitalisation plan (2017-10-24)",
    ("PNB", "2017-10-25"): "Government PSU-bank recapitalisation plan (2017-10-24)",
    ("IBULHSGFIN", "2019-09-30"): "Delhi High Court agreed to hear a PIL alleging fund diversion; ended 34% lower",
    ("IBULHSGFIN", "2020-03-19"): "COVID-19 crash week: housing-finance sell-off (no corporate action)",
    ("IBULHSGFIN", "2020-06-19"): "Bulk deals, Morgan Stanley bought a 1.06% stake (2020-06-19)",
    ("IDEA", "2019-07-29"): "Q1 FY20 results: subscriber and revenue decline",
    ("IDEA", "2019-11-19"): "Tariff increase announced after hours on 2019-11-18",
    ("IDEA", "2020-02-19"): "Reports that the government would not invoke bank guarantees over AGR dues",
    ("IDEA", "2020-03-13"): "Hopes of a government relief package for telecoms",
    ("IDEA", "2020-03-18"): "Supreme Court rejected self-assessment of AGR dues",
    ("BANDHANBNK", "2020-03-23"): _COVID,
    ("BANDHANBNK", "2020-03-26"): "Rebound of beaten-down bank stocks at the March 2020 F&O expiry",
    ("RECLTD", "2024-06-04"): "General-election results day: sell-off in state-owned stocks (Nifty Next 50 -9.3%)",
}
NSE_CHECK_TOL = 0.015   # |Yahoo return - NSE close/prevclose return| allowed (dividends, rounding)

if load()["data"]["universe"] == "next50":
    CANDIDATES = {**CANDIDATES, **CANDIDATES_NEXT50}
    DEMERGERS = DEMERGERS + DEMERGERS_NEXT50
    KNOWN_EVENTS = {**KNOWN_EVENTS, **KNOWN_EVENTS_NEXT50}


CHECK_NSE = load()["data"]["universe"] == "next50"


def nse_check(sym: str, day: pd.Timestamp, r_vendor: float) -> dict:
    """Protocol v2: a flagged day is accepted only with a reviewed explanation AND an NSE bhavcopy
    return (close / previous close - 1) within NSE_CHECK_TOL of the vendor return."""
    from data import nse_bhav
    al = pd.read_csv(ROSTER / "symbol_aliases.csv", dtype=str)
    iv = pd.read_csv(ROSTER / "membership_intervals.csv", dtype=str)
    names = set(al.loc[al["symbol"] == sym, "alias"]) | {sym}
    isins = set(iv.loc[iv["symbol"] == sym, "isin"].dropna())
    row = nse_bhav.row(day, isins, names)
    r_nse = row["close"] / row["prevclose"] - 1 if row else float("nan")
    agree = row is not None and abs(r_vendor - r_nse) <= NSE_CHECK_TOL
    why = KNOWN_EVENTS.get((sym, day.strftime("%Y-%m-%d")), "")
    return {"symbol": sym, "date": day.date(), "r_vendor": round(r_vendor, 4),
            "r_nse": round(r_nse, 4) if row else None, "nse_open": row["open"] if row else None,
            "nse_close": row["close"] if row else None, "nse_prevclose": row["prevclose"] if row else None,
            "nse_agrees": agree, "explanation": why, "accepted": bool(agree and why),
            "bhavcopy": row["url"] if row else ""}


def candidates(symbol: str) -> list[str]:
    return CANDIDATES.get(symbol, [f"{symbol}.NS"])


def fetch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Cached wide frames; only tickers missing from the cache are downloaded."""
    cfg = load()["data"]
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {f: RAW / f"yahoo_{f.replace(' ', '_').lower()}.csv" for f in yahoo.FIELDS}
    have = {f: pd.read_csv(p, index_col="Date", parse_dates=True) for f, p in paths.items() if p.exists()}
    wanted = sorted({t for s in symbols for t in candidates(s)})
    cached = set(have["Adj Close"].columns) if "Adj Close" in have else set()
    missing = [t for t in wanted if t not in cached]
    if missing:
        end = (date.fromisoformat(cfg["freeze"]) + timedelta(days=1)).isoformat()
        new = yahoo.download(missing, cfg["price_start"], end, chunk=15)
        for f, p in paths.items():
            parts = [x for x in (have.get(f), new.get(f)) if x is not None and not x.empty]
            have[f] = pd.concat(parts, axis=1) if parts else pd.DataFrame()
            have[f].index.name = "Date"
            have[f].to_csv(p)
    return have


def splits_table(tickers: list[str]) -> pd.DataFrame:
    """Yahoo-recorded splits/bonuses (cached in data/raw/prices/yahoo_splits.csv)."""
    import time

    import yfinance as yf
    path = RAW / "yahoo_splits.csv"
    tab = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ticker", "date", "factor"])
    for tk in [t for t in tickers if t not in set(tab["ticker"])]:
        try:
            sp = yf.Ticker(tk).splits
            rows = [{"ticker": tk, "date": pd.Timestamp(d).tz_localize(None).date(), "factor": float(f)}
                    for d, f in sp.items()]
        except Exception:  # noqa: BLE001  (record the attempt; no splits known)
            rows = []
        tab = pd.concat([tab, pd.DataFrame(rows or [{"ticker": tk, "date": None, "factor": None}])])
        time.sleep(1.5)
    tab.to_csv(path, index=False)
    return tab.dropna(subset=["date"])


def repair_split_artifacts(sym: str, tk: str, px: pd.Series, splits: pd.DataFrame) -> tuple[pd.Series, list]:
    """Yahoo sometimes back-adjusts a split only to the start of its calendar year,
    leaving a fake drop of 1/f - 1 at that boundary. Find such a day near each recorded
    split and rescale everything before it by 1/f."""
    logs = []
    r = px.pct_change(fill_method=None)
    for s in splits[splits["ticker"] == tk].itertuples():
        d, f = pd.Timestamp(s.date), float(s.factor)
        near = r[(r.index >= d - pd.Timedelta(days=400)) & (r.index <= d + pd.Timedelta(days=10))]
        hits = near[((1 + near) * f - 1).abs() < 0.03]
        hits = hits[hits.abs() > 0.15]
        for x in hits.index:
            before = px.index < x
            px = px.copy()
            px[before] = px[before] / f
            true_r = float(px[x] / px[px.index < x].dropna().iloc[-1] - 1)
            logs.append({"symbol": sym, "ticker": tk, "date": x.date(), "kind": "split_boundary_artifact",
                         "detail": f"split {f:g} on {d.date()}; raw return {near[x]:+.2%} -> {true_r:+.2%}",
                         "source": "Yahoo split record (data/raw/prices/yahoo_splits.csv)"})
            r = px.pct_change(fill_method=None)
    return px, logs


def repair_demergers(sym: str, tk: str, px: pd.Series, close: pd.Series,
                     splits: pd.DataFrame) -> tuple[pd.Series, list]:
    logs = []
    for s, ex, disc, prev, src in DEMERGERS:
        if s != sym:
            continue
        ex = pd.Timestamp(ex)
        later = splits[(splits["ticker"] == tk) & (pd.to_datetime(splits["date"]) > ex)]["factor"]
        scale = float(np.prod(later)) if len(later) else 1.0
        raw_close_ex = float(close[ex]) * scale
        true_r = raw_close_ex / disc - 1.0
        prev_idx = px.index[px.index < ex][-1]
        was = float(px[ex] / px[prev_idx] - 1)
        k = (px[ex] / (1 + true_r)) / px[prev_idx]
        px = px.copy()
        px[px.index < ex] *= k
        logs.append({"symbol": sym, "ticker": tk, "date": ex.date(), "kind": "demerger",
                     "detail": f"discovered {disc} vs prev close {prev}; ex-date return {was:+.2%} -> {true_r:+.2%}",
                     "source": src})
    return px, logs


def windows(iv: pd.DataFrame, sessions: pd.DatetimeIndex, warmup: int) -> dict[str, np.ndarray]:
    """Boolean session mask per symbol: membership windows plus `warmup` sessions before."""
    out = {}
    for sym, g in iv.groupby("symbol"):
        m = np.zeros(len(sessions), bool)
        for r in g.itertuples():
            i0 = int(np.searchsorted(sessions, r.start))
            i1 = int(np.searchsorted(sessions, r.end_exclusive))
            m[max(0, i0 - warmup):i1] = True
        out[sym] = m
    return out


def build():
    cfg = load()["data"]
    win = load()["portfolio"]["cov_window"]
    iv = pd.read_csv(ROSTER / "membership_intervals.csv", parse_dates=["start", "end_exclusive"])
    symbols = sorted(iv["symbol"].unique())
    frames = fetch(symbols)
    adj, close = frames["Adj Close"], frames["Close"]
    mk = pd.read_parquet(PROC / "market.parquet")
    S = mk.index
    masks = windows(iv, S, win)
    member_only = windows(iv, S, 0)

    chosen, qa, repairs, jump_checks = {}, [], [], []
    picks = {}
    for sym in symbols:
        best = None
        for tk in candidates(sym):
            if tk not in adj.columns:
                continue
            px = adj[tk].reindex(S)
            cov = float(px[masks[sym]].notna().mean())
            if best is None or cov > best[1]:
                best = (tk, cov, px)
        picks[sym] = best
    splits = splits_table(sorted({b[0] for b in picks.values() if b is not None}))

    for sym in symbols:
        best = picks[sym]
        if best is None or best[1] == 0.0:
            qa.append({"symbol": sym, "ticker": "", "coverage": 0.0, "member_coverage": 0.0,
                       "jumps": "", "status": "excluded", "reason": "no Yahoo series (NSE or BSE)"})
            continue
        tk, cov, px = best
        px, l1 = repair_split_artifacts(sym, tk, px, splits)
        px, l2 = repair_demergers(sym, tk, px, close[tk].reindex(S), splits)
        repairs += l1 + l2
        r = px.pct_change(fill_method=None)
        mcov = float(px[member_only[sym]].notna().mean())
        in_win = pd.Series(member_only[sym], index=S)
        big = r[(r.abs() > cfg["jump_flag"]) & in_win]
        unexplained = [d for d in big.index if (sym, d.strftime("%Y-%m-%d")) not in KNOWN_EVENTS]
        if CHECK_NSE:
            for d in big.index:
                chk = nse_check(sym, d, float(r[d]))
                jump_checks.append(chk)
                if not chk["accepted"] and d not in unexplained:
                    unexplained.append(d)
        status, reason = "adjusted", ""
        if mcov < cfg["min_coverage"]:
            status, reason = "excluded", f"member-window coverage {mcov:.1%}"
        elif unexplained:
            status, reason = "excluded", "unexplained jump(s)"
        qa.append({"symbol": sym, "ticker": tk, "coverage": round(cov, 4),
                   "member_coverage": round(mcov, 4),
                   "jumps": " ".join(f"{d.date()}:{r[d]:+.1%}" for d in big.index),
                   "status": status, "reason": reason})
        chosen[sym] = px
    qa = pd.DataFrame(qa)
    PROC.mkdir(parents=True, exist_ok=True)
    qa.to_csv(PROC / "price_qa.csv", index=False)
    pd.DataFrame(repairs).to_csv(PROC / "price_repairs.csv", index=False)
    if CHECK_NSE:
        pd.DataFrame(jump_checks).to_csv(PROC / "price_jump_checks.csv", index=False)
    explained = pd.DataFrame([{"symbol": s, "date": d, "reason": why} for (s, d), why in KNOWN_EVENTS.items()])
    explained.to_csv(PROC / "price_jumps_explained.csv", index=False)
    panel = pd.DataFrame(chosen, index=S)
    panel.index.name = "Date"
    panel.to_parquet(PROC / "prices_adjusted.parquet")
    return qa, panel


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 90)
    qa, _ = build()
    print(qa["status"].value_counts().to_string())
    print(qa[(qa["status"] != "adjusted") | (qa["jumps"] != "")].to_string(index=False))
