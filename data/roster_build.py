"""Daily point-in-time Nifty 50 membership, validated against every archived snapshot.

Start from the 2013-10-13 snapshot, apply the dated changes in data/roster/changes.csv
(each with its NSE notice or news source), and expand to NSE sessions. Symbols are
canonicalised by ISIN to the latest symbol NSE printed for that company, so a ticker
rename (SSLT->VEDL, TATAMOTORS->TMPV) is not an index change.

Documented exceptions (reported, never silently ignored):
- demerger placeholder constituents (JIOFIN Jul-Aug 2023, ITC Hotels Jan 2025, Tata
  Motors CV Oct-Nov 2025) are excluded: they carry no tradable price while placeholders;
- Tata Motors DVR was an additional security from 2016-04-01 to 2017-09-28, so the
  index held 51 securities (50 companies) in that window;
- archived copies of the pre-rename file that NSE had not yet updated.

Run:  python -m data.roster_build
"""
from __future__ import annotations

import pandas as pd

from config import ROOT, load
from data.roster import OUT_DIR, membership_sets

DVR_WINDOW = (pd.Timestamp("2016-04-01"), pd.Timestamp("2017-09-28"))
KNOWN_STALE = {
    pd.Timestamp("2015-05-19"): "pre-rename ind_niftylist.csv not yet updated for the 27 Mar 2015 change",
}


def canonical_map() -> tuple[dict[str, str], dict[str, str]]:
    """symbol -> canonical symbol (latest NSE symbol for its ISIN), and canonical -> ISIN."""
    m = pd.read_csv(OUT_DIR / "snapshot_members.csv", dtype=str)
    idx = pd.read_csv(OUT_DIR / "snapshot_index.csv", dtype={"capture_ts": str}, parse_dates=["date"])
    m = m.merge(idx[["capture_ts", "date"]].drop_duplicates("capture_ts"), on="capture_ts")
    # Equity ISINs only (security-type digits "01"); the JSON API sometimes carries the
    # ISIN of a bond or partly-paid line for the same company.
    m = m[(m["isin"].str.len() == 12) & (m["isin"].str[7:9] == "01")].sort_values("date")
    latest = m.groupby("isin")["symbol"].last()
    canon = {s: latest[i] for s, i in m[["symbol", "isin"]].drop_duplicates().itertuples(index=False)}
    isin_of = {latest[i]: i for i in latest.index}
    return canon, isin_of


def sessions() -> pd.DatetimeIndex:
    cfg = load()["data"]
    mk = pd.read_parquet(ROOT / "data" / "processed" / "market.parquet")
    s = mk.index
    return s[(s >= pd.Timestamp(cfg["study_start"])) & (s <= pd.Timestamp(cfg["freeze"]))]


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canon, isin_of = canonical_map()
    runs, sets = membership_sets()
    anchor = {canon.get(s, s) for s in sets[runs.iloc[0]["members_hash"]]}
    ch = pd.read_csv(OUT_DIR / "changes.csv", parse_dates=["effective_date"])
    ch = ch[ch["action"].isin(["add", "remove"])].copy()
    ch["symbol"] = ch["symbol"].map(lambda s: canon.get(s, s))
    S = sessions()

    current = set(anchor)
    open_at = {s: S[0] for s in current}
    intervals = []
    for d, g in ch.groupby("effective_date", sort=True):
        eff = S[S >= d][0]  # first session of the new composition
        for s in g.loc[g["action"] == "remove", "symbol"]:
            if s not in current:
                raise ValueError(f"{d.date()}: removing non-member {s}")
            intervals.append((s, open_at.pop(s), eff))
            current.remove(s)
        for s in g.loc[g["action"] == "add", "symbol"]:
            if s in current:
                raise ValueError(f"{d.date()}: adding existing member {s}")
            current.add(s)
            open_at[s] = eff
    end = S[-1] + pd.Timedelta(days=1)
    intervals += [(s, t0, end) for s, t0 in open_at.items()]
    iv = pd.DataFrame(intervals, columns=["symbol", "start", "end_exclusive"])
    iv["isin"] = iv["symbol"].map(isin_of)
    iv = iv.sort_values(["symbol", "start"]).reset_index(drop=True)

    syms = sorted(iv["symbol"].unique())
    daily = pd.DataFrame(False, index=S, columns=syms)
    for r in iv.itertuples():
        daily.loc[(S >= r.start) & (S < r.end_exclusive), r.symbol] = True

    # Validation against every snapshot on or after the study start.
    idx = pd.read_csv(OUT_DIR / "snapshot_index.csv", parse_dates=["date"], dtype={"capture_ts": str})
    mem = pd.read_csv(OUT_DIR / "snapshot_members.csv", dtype=str)
    snap_sets = {h: {canon.get(s, s) for s in g["symbol"]} for h, g in mem.groupby("members_hash")}
    rows = []
    for r in idx.drop_duplicates(["date", "members_hash"]).itertuples():
        if r.date < S[0]:
            continue
        on = S[S <= r.date][-1]
        built = set(daily.columns[daily.loc[on]])
        if on >= DVR_WINDOW[0] and on <= DVR_WINDOW[1]:
            built.discard("TATAMTRDVR")
        snap = snap_sets[r.members_hash]
        ok = built == snap
        rows.append({"snapshot_date": r.date.date(), "session": on.date(), "source": r.source,
                     "match": ok, "missing_in_build": " ".join(sorted(snap - built)),
                     "extra_in_build": " ".join(sorted(built - snap)),
                     "explanation": "" if ok else KNOWN_STALE.get(r.date, "UNEXPLAINED")})
    val = pd.DataFrame(rows)
    counts = daily.sum(axis=1)
    expected = pd.Series(50, index=S)
    expected[(S >= DVR_WINDOW[0]) & (S <= DVR_WINDOW[1])] = 51
    bad_counts = counts[counts != expected]

    iv.to_csv(OUT_DIR / "membership_intervals.csv", index=False)
    daily.to_parquet(OUT_DIR / "membership_daily.parquet")
    val.to_csv(OUT_DIR / "validation.csv", index=False)
    return iv, val, bad_counts


if __name__ == "__main__":
    iv, val, bad = build()
    print(f"symbols ever in the index (2014-01-01..freeze): {iv['symbol'].nunique()}")
    print(f"membership intervals: {len(iv)}")
    print(f"snapshots checked: {len(val)}, matching: {int(val['match'].sum())}")
    mism = val[~val["match"]]
    if len(mism):
        print(mism.to_string(index=False))
    print(f"sessions with an unexpected member count: {len(bad)}")
    if len(bad):
        print(bad.head(20).to_string())
