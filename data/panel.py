"""Assemble the point-in-time panel and the engine arrays.

Inputs: market.parquet, prices_adjusted.parquet, price_qa.csv, membership_daily.parquet,
rf.parquet. Output: data/processed/engine.npz (everything env.engine needs), so training
workers load it in milliseconds instead of recomputing risk-parity weights.

Run:  python -m data.panel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROOT, load
from env.build import assemble
from env.engine import MarketData

PROC = ROOT / "data" / "processed"
ROSTER = ROOT / "data" / "roster"
ENGINE = PROC / "engine.npz"


def inputs():
    mk = pd.read_parquet(PROC / "market.parquet")
    S = mk.index
    px = pd.read_parquet(PROC / "prices_adjusted.parquet").reindex(S)
    qa = pd.read_csv(PROC / "price_qa.csv")
    mem = pd.read_parquet(ROSTER / "membership_daily.parquet")
    symbols = sorted(mem.columns)
    usable = set(qa.loc[qa["status"] == "adjusted", "symbol"])
    px = px.reindex(columns=symbols)
    px[[s for s in symbols if s not in usable]] = np.nan  # excluded: never eligible
    member = mem.reindex(index=S, columns=symbols).fillna(False).astype(bool)
    rf = pd.read_parquet(PROC / "rf.parquet").reindex(S)["accrual"].to_numpy()
    R = px.pct_change(fill_method=None).to_numpy()
    return S, mk, R, member.to_numpy(), rf, symbols


def build() -> MarketData:
    cfg = load()
    S, mk, R, member, rf, symbols = inputs()
    data = assemble(S, mk[["nsei", "vix", "usdinr"]], R, member, rf,
                    window=cfg["portfolio"]["cov_window"], symbols=symbols)
    np.savez_compressed(ENGINE, dates=S.values.astype("datetime64[ns]").astype(np.int64), R=data.R,
                        W=data.W, exit_mask=data.exit_mask, delist_mask=data.delist_mask, rf=data.rf,
                        feats=data.feats, rp_ret=data.rp_ret, eligible=data.eligible,
                        member=member, symbols=np.array(symbols))
    return data


def load_engine() -> MarketData:
    z = np.load(ENGINE, allow_pickle=False)
    data = MarketData(dates=pd.DatetimeIndex(z["dates"].astype("datetime64[ns]")), R=z["R"], W=z["W"],
                      exit_mask=z["exit_mask"], delist_mask=z["delist_mask"], rf=z["rf"],
                      feats=z["feats"], rp_ret=z["rp_ret"], symbols=list(z["symbols"]))
    data.eligible = z["eligible"]
    data.member = z["member"]
    return data


if __name__ == "__main__":
    d = build()
    S = d.dates
    elig = d.eligible.sum(axis=1)
    mem = d.member if hasattr(d, "member") else None
    yr = pd.Series(elig, index=S)
    print(f"sessions {len(S)}  {S[0].date()}..{S[-1].date()}  symbols {len(d.symbols)}")
    print("eligible stocks per session (yearly min/median/max, 2014+):")
    print(yr["2014":].groupby(yr["2014":].index.year).agg(["min", "median", "max"]).to_string())
    print(f"forced exits: {int(d.exit_mask.sum())}, delisting exits: {int(d.delist_mask.sum())}")
    w = pd.Series(d.W[S.get_loc(S[S <= '2026-11-06'][-1])], index=d.symbols)
    print("last-session risk-parity weights, top 5:", w.sort_values(ascending=False).head(5).round(4).to_dict())
