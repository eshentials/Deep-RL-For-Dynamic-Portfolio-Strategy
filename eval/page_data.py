"""Daily series for the shared results page (charts of growth, drawdown and VIX).

Writes results/page/daily.csv from the protocol test returns, the post-hoc VIX rule's
returns and India VIX: one row per session of the test path. Values are copied, not
recomputed (returns to 10 significant digits).

Run:  python -m eval.page_data
"""
from __future__ import annotations

import pandas as pd

from config import ROOT

RES = ROOT / "results"
COLS = ["ppo_ensemble", "vol_target_floored", "equal_weight", "niftybees", "rp_fixed_0.7"]


def main():
    r = pd.read_parquet(RES / "test_returns.parquet")
    out = r[COLS + ["_rf"]].rename(columns={"_rf": "rf"})
    vr = pd.read_parquet(RES / "exploratory" / "vix_rule" / "returns.parquet")["vix_rule"]
    if not vr.index.equals(out.index):
        raise SystemExit("VIX rule returns are on a different path")
    out.insert(len(COLS), "vix_rule", vr)
    vix = pd.read_parquet(ROOT / "data" / "processed" / "market.parquet")["vix"].reindex(out.index)
    out["india_vix"] = vix.round(2)
    out.index.name = "date"
    (RES / "page").mkdir(exist_ok=True)
    out.to_csv(RES / "page" / "daily.csv", float_format="%.10g", date_format="%Y-%m-%d")
    print(f"wrote results/page/daily.csv: {len(out)} rows, columns {list(out.columns)}")


if __name__ == "__main__":
    main()
