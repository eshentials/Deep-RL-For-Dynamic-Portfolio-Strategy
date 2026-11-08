"""Daily series for the shared results page (charts of growth, drawdown and VIX).

Protocol v1: results/page/daily.csv from the test returns, the post-hoc VIX rule's
returns and India VIX. Protocol v2: studies/next50/results/page/daily.csv from its test
returns (the VIX rule is a registered strategy there) and India VIX. One row per session
of the test path; values are copied, not recomputed (returns to 10 significant digits).

Run:  python -m eval.page_data                                  # v1
      RB_PROTOCOL=protocol_v2.yaml python -m eval.page_data     # v2
"""
from __future__ import annotations

import pandas as pd

from config import ROOT, load, study_root

RES = ROOT / "results"
COLS = ["ppo_ensemble", "vol_target_floored", "equal_weight", "niftybees", "rp_fixed_0.7"]


def main():
    if load()["version"] != 1:
        return main_v2()
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


def main_v2():
    res = study_root() / "results"
    cols = ["ppo_ensemble", "vol_target_floored", "vix_rule", "equal_weight", load()["data"]["benchmark_key"]]
    r = pd.read_parquet(res / "test_returns.parquet")
    out = r[cols + ["_rf"]].rename(columns={"_rf": "rf"})
    vix = pd.read_parquet(study_root() / "data" / "processed" / "market.parquet")["vix"].reindex(out.index)
    out["india_vix"] = vix.round(2)
    out.index.name = "date"
    (res / "page").mkdir(exist_ok=True)
    out.to_csv(res / "page" / "daily.csv", float_format="%.10g", date_format="%Y-%m-%d")
    print(f"wrote {(res / 'page' / 'daily.csv').relative_to(ROOT)}: {len(out)} rows, columns {list(out.columns)}")


if __name__ == "__main__":
    main()
