"""Post-unblinding analyses (protocol: "Any analysis after the test run is labelled
exploratory and logged in ledger.csv"). Nothing here is pre-registered, and nothing
here changes the H1 verdict in results/report.md. These re-read the test-run returns;
no strategy is re-run or re-selected.

1. Period splits: the H1 comparison and a few others on 2019-2021 (years not seen
   before this study), 2022-2026 (previously seen) and all years except 2020.
2. The three deepest NIFTYBEES drawdown episodes on the whole path, with no
   restriction on the year of the peak.
3. What the ensemble chose, grouped by the India VIX and the Nifty drawdown it saw.

Run:  python -m eval.exploratory   (writes results/exploratory/, appends a ledger row)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROOT, load
from eval import metrics as M
from eval.bootstrap import paired_sharpe_diff

RES = ROOT / "results"
OUT = RES / "exploratory"
COMPARE = {"vol_target_floored": "Vol targeting (floor 0.5)",
           "rp_fixed_0.7": "Risk parity, e=0.70 (nearest fixed exposure to the ensemble's 0.66 average)",
           "equal_weight": "Equal weight (monthly)", "niftybees": "NIFTYBEES buy-and-hold"}
MAIN = ["ppo_ensemble", "vol_target_floored", "rp_fixed_0.7", "equal_weight", "niftybees"]


def period_splits(rets, rf, b):
    yr = rets.index.year
    periods = {"2019-2021 (not previously seen)": yr <= 2021,
               "2022-2026 (previously seen)": yr >= 2022,
               "All years except 2020": yr != 2020}
    rows = []
    for name, m in periods.items():
        for c, lab in COMPARE.items():
            d = paired_sharpe_diff(rets.loc[m, "ppo_ensemble"], rets.loc[m, c], rf[m],
                                   block=b["block"], draws=b["draws"], seed=b["seed"])
            rows.append({"period": name, "vs": lab, "sessions": int(m.sum()),
                         "sharpe_ppo": M.sharpe(rets.loc[m, "ppo_ensemble"].to_numpy(), rf[m]),
                         "sharpe_other": M.sharpe(rets.loc[m, c].to_numpy(), rf[m]),
                         "delta": d["observed"], "p05": d["p05"], "ci_lo": d["ci95"][0],
                         "ci_hi": d["ci95"][1], "share_le_0": d["share_le_0"]})
    return pd.DataFrame(rows)


def deepest_episodes(nav: pd.Series, k: int = 3):
    """Non-overlapping peak-to-trough episodes (an episode runs from a running high
    until the next new high), deepest first."""
    peak = nav.cummax()
    new_high = nav >= peak
    eps, start = [], None
    for d in nav.index:
        if new_high[d]:
            start = d
        else:
            depth = nav[d] / nav[start] - 1
            if not eps or eps[-1][0] != start:
                eps.append([start, d, depth])
            elif depth < eps[-1][2]:
                eps[-1][1:] = [d, depth]
    eps.sort(key=lambda e: e[2])
    return eps[:k]


def episodes(rets):
    navs = {c: pd.Series(M.nav(rets[c].to_numpy())[1:], index=rets.index) for c in MAIN}
    rows = []
    for rank, (p, t, depth) in enumerate(deepest_episodes(navs["niftybees"]), 1):
        for c in MAIN:
            seg = navs[c][p:t]
            rows.append({"rank": rank, "peak": p.date(), "trough": t.date(), "strategy": c,
                         "return_peak_to_trough": float(seg.iloc[-1] / seg.iloc[0] - 1),
                         "max_dd_in_window": float((seg / seg.cummax() - 1).min())})
    return pd.DataFrame(rows)


def exposure_by_regime():
    dec = pd.read_csv(RES / "ensemble_decisions.csv", parse_dates=["date"]).set_index("date")
    mk = pd.read_parquet(ROOT / "data" / "processed" / "market.parquet")
    nifty_dd = mk["nsei"] / mk["nsei"].rolling(252, min_periods=1).max() - 1
    df = dec.assign(vix=mk["vix"].reindex(dec.index), nifty_dd=nifty_dd.reindex(dec.index))
    df["vix_bucket"] = pd.cut(df["vix"], [0, 15, 20, 25, np.inf], right=False,
                              labels=["<15", "15-20", "20-25", ">=25"])
    df["dd_bucket"] = pd.cut(df["nifty_dd"], [-np.inf, -0.20, -0.10, -0.05, 0.0001], right=False,
                             labels=["< -20%", "-20% to -10%", "-10% to -5%", "-5% to 0%"])
    def by(col):
        g = df.groupby(col, observed=False)["exposure"]
        return pd.DataFrame({"decisions": g.size(), "mean_exposure": g.mean(),
                             "share_at_floor": g.apply(lambda s: (s <= 0.5 + 1e-9).mean() if len(s) else np.nan)}
                            ).reset_index()
    return by("vix_bucket"), by("dd_bucket"), df[["exposure", "vix", "nifty_dd"]].corr().loc["exposure"]


def _md_table(df, cols, fmt):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.get(c, str)(getattr(r, c)) for c in cols) + " |")
    return lines


def main():
    cfg = load()
    b = cfg["evaluation"]["bootstrap"]
    rets = pd.read_parquet(RES / "test_returns.parquet")
    rf = rets.pop("_rf").to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    sp = period_splits(rets, rf, b)
    ep = episodes(rets)
    vix, dd, corr = exposure_by_regime()
    sp.to_csv(OUT / "period_splits.csv", index=False)
    ep.to_csv(OUT / "deepest_episodes.csv", index=False)
    vix.to_csv(OUT / "exposure_by_vix.csv", index=False)
    dd.to_csv(OUT / "exposure_by_nifty_drawdown.csv", index=False)

    f3 = lambda x: f"{x:+.3f}"  # noqa: E731
    pct = lambda x: f"{x * 100:+.1f}%"  # noqa: E731
    lines = ["# Exploratory analyses (post-unblinding, not pre-registered)", "",
             "These re-read the protocol test run's returns. They are not part of H1 and do not change "
             "its verdict. Logged in ledger.csv as kind `exploratory`.", "",
             "## 1. Period splits (paired block bootstrap, same settings as H1)", ""]
    sp_fmt = sp.assign(sharpe_ppo=sp.sharpe_ppo.map("{:.3f}".format), sharpe_other=sp.sharpe_other.map("{:.3f}".format),
                       ci=[f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(sp.ci_lo, sp.ci_hi)])
    lines += _md_table(sp_fmt, ["period", "vs", "sharpe_ppo", "sharpe_other", "delta", "p05", "ci", "share_le_0"],
                       {"delta": f3, "p05": f3, "share_le_0": "{:.3f}".format})
    lines += ["", "## 2. Deepest NIFTYBEES drawdown episodes on the whole path", ""]
    lines += _md_table(ep, ["rank", "peak", "trough", "strategy", "return_peak_to_trough", "max_dd_in_window"],
                       {"return_peak_to_trough": pct, "max_dd_in_window": pct})
    lines += ["", "## 3. Ensemble exposure by the market state at the decision", "",
              f"Correlation of chosen exposure with India VIX: {corr['vix']:+.2f}; with the Nifty drawdown "
              f"from its 252-session high: {corr['nifty_dd']:+.2f} (negative = more exposure in deeper drawdowns).", ""]
    two = lambda x: "" if pd.isna(x) else f"{x:.2f}"  # noqa: E731
    lines += _md_table(vix, ["vix_bucket", "decisions", "mean_exposure", "share_at_floor"],
                       {"mean_exposure": two, "share_at_floor": two})
    lines += [""]
    lines += _md_table(dd, ["dd_bucket", "decisions", "mean_exposure", "share_at_floor"],
                       {"mean_exposure": two, "share_at_floor": two})
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    from train import ledger
    ledger.append("exploratory", "post-unblinding analyses of the protocol test returns: period splits "
                  "(2019-21, 2022-26, ex-2020), deepest drawdown episodes, exposure by VIX/drawdown",
                  folds=" ".join(map(str, cfg["folds"]["test_years"])), touched_test=True,
                  counts_as_trial=False, notes="results/exploratory/; no strategy re-run or re-selected")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
