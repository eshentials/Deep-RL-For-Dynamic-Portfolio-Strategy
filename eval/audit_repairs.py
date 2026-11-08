"""Post-test audit of Study 2's price repairs against NSE's daily files (protocol v2).

The manuscript's appendix showed that the automatic split-boundary repair inherited from
protocol v1 also fired on CONCOR on 2024-06-04, a genuine election-day fall whose size happened
to match a later split factor. This audit checks every repair in
studies/next50/data/processed/price_repairs.csv against the exchange's own close and previous
close (open = special pre-open price for demergers), and whether the symbol could have carried
weight on or after the date: a wrong return at d enters covariance windows of sessions d..d+119.

It reads the registered data and writes only studies/next50/results/repair_audit.csv; the
registered data and their manifest are not touched. Logged in ledger.csv before it runs.

Run:  RB_PROTOCOL=protocol_v2.yaml python -m eval.audit_repairs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import load, require_version, study_root
from data import nse_bhav

TOL = 0.015


def main():
    require_version(2, "eval.audit_repairs")
    base = study_root()
    rep = pd.read_csv(base / "data" / "processed" / "price_repairs.csv")
    px = pd.read_parquet(base / "data" / "processed" / "prices_adjusted.parquet")
    raw = pd.read_csv(base / "data" / "raw" / "prices" / "yahoo_adj_close.csv", index_col=0, parse_dates=True)
    z = np.load(base / "data" / "processed" / "engine.npz", allow_pickle=False)
    syms, dates = list(z["symbols"]), pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"))
    window = load()["portfolio"]["cov_window"]
    al = pd.read_csv(base / "data" / "roster" / "symbol_aliases.csv")
    rows = []
    for r in rep.itertuples():
        d = pd.Timestamp(r.date)
        url = nse_bhav.url_for(d)
        b = nse_bhav._download(url)
        df = nse_bhav._parse(b, url) if b is not None else pd.DataFrame()
        names = set(al.loc[al["symbol"] == r.symbol, "alias"]) | {r.symbol}
        m = df[df["symbol"].isin(names)] if len(df) else df
        nse = m.iloc[0] if len(m) else None
        prev = px.index[px.index < d][-1]
        r_rep = float(px.loc[d, r.symbol] / px.loc[prev, r.symbol] - 1)
        rr = raw[r.ticker].dropna()
        r_raw = float(rr.loc[d] / rr[rr.index < d].iloc[-1] - 1)
        if r.kind == "demerger":   # the exchange's reference is the special pre-open price (the open)
            r_nse = float(nse["close"] / nse["open"] - 1) if nse is not None else np.nan
        else:
            r_nse = float(nse["close"] / nse["prevclose"] - 1) if nse is not None else np.nan
        i, t = syms.index(r.symbol), dates.get_loc(d)
        exposed = bool(z["eligible"][t:t + window, i].any())
        rows.append({"symbol": r.symbol, "date": d.date(), "kind": r.kind, "r_vendor_raw": round(r_raw, 4),
                     "r_repaired": round(r_rep, 4), "r_nse": round(r_nse, 4),
                     "repair_correct": bool(abs(r_rep - r_nse) <= TOL),
                     "eligible_within_window": exposed, "bhavcopy": url})
    out = pd.DataFrame(rows)
    out.to_csv(base / "results" / "repair_audit.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
