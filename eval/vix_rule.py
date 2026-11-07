"""Post-hoc baseline: a VIX threshold rule (EXPLORATORY, not pre-registered).

Why it exists: in the protocol test run the PPO ensemble's exposure rose with India
VIX (correlation +0.62, results/exploratory/). Reviewers will ask whether PPO is more
than "half exposure in calm markets, full exposure after a VIX spike". This rule is
that sentence, fixed here before it was run on any test year (2026-11-07):

    exposure = 1.0 if India VIX at close t >= threshold, else 0.5; horizon h sessions
    grid: threshold in {15, 20, 25} x h in {5, 21}  (6 points, the size of the vol-targeting grid)
    selection: per fold, mean Sharpe on its two validation years from cash, 10 bp costs
    test: the protocol engine, path and year-boundary rule, unchanged

The ledger row is written before any test-year score is computed, as kind
`sensitivity` (a new strategy variant), so it counts as a trial in the Deflated Sharpe.
Outputs: results/exploratory/vix_rule/.

Run:  python -m eval.vix_rule
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from baselines import rules
from baselines.select import _score
from config import ROOT, load
from env.engine import run
from env.features import MARKET_FEATURES
from eval import metrics as M
from eval.bootstrap import paired_sharpe_diff
from eval.dsr import dsr
from train.common import FoldSwitch, folds

RES = ROOT / "results"
OUT = RES / "exploratory" / "vix_rule"
VIX_COL = MARKET_FEATURES.index("vix")
THRESHOLDS = [0.15, 0.20, 0.25]  # India VIX / 100, as stored in the feature array
HORIZONS = [5, 21]
LOW, HIGH = 0.5, 1.0
NAMES = {"ppo_ensemble": "PPO ensemble", "vix_rule": "VIX threshold rule (post-hoc)",
         "vol_target_floored": "Vol targeting (floor 0.5)", "rp_fixed_0.7": "Risk parity, e=0.70"}


def vix_threshold(data, threshold: float, horizon: int, low: float = LOW, high: float = HIGH):
    v = data.feats[:, VIX_COL]

    def pol(sim, t):
        return (high if v[t] >= threshold else low), horizon
    return pol


def select(data, fold, cost_bps: float) -> dict:
    grid = []
    for th, hz in itertools.product(THRESHOLDS, HORIZONS):
        grid.append(({"threshold": th, "horizon": hz},
                     _score(data, lambda th=th, hz=hz: vix_threshold(data, th, hz), fold, cost_bps)))
    best = max(grid, key=lambda g: g[1])
    return {"params": best[0], "val_sharpe": best[1], "grid": grid}


def session_exposure(dec: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Exposure set at each decision close, carried forward to later sessions."""
    s = pd.Series(dec["exposure"].to_numpy(), index=pd.DatetimeIndex(dec["date"]))
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def _table(df: pd.DataFrame, cols: list[str], fmt: dict) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(fmt.get(c, str)(getattr(r, c)) for c in cols) + " |")
    return out


def main():
    cfg = load()
    cost = cfg["portfolio"]["cost_bps"]
    b = cfg["evaluation"]["bootstrap"]
    years = cfg["folds"]["test_years"]
    from data.panel import load_engine
    from train import ledger
    data = load_engine()

    ledger.append("sensitivity", "POST-HOC exploratory baseline (not pre-registered): VIX threshold rule, "
                  "exposure 1.0 if India VIX >= threshold else 0.5; threshold {15,20,25} x horizon {5,21} "
                  "selected per fold on validation", folds=" ".join(map(str, years)), touched_test=True,
                  counts_as_trial=True, notes="results/exploratory/vix_rule/; logged before evaluation; "
                  "counted as a trial (conservative)")

    fl = {f.test_year: f for f in folds(data, years)}
    sel = {y: select(data, fl[y], cost) for y in years}
    start, end = data.last_session_before(min(years)), data.last_session_of(max(years))
    res = run(data, FoldSwitch(data, {y: vix_threshold(data, **sel[y]["params"]) for y in years}),
              start, end, cost_bps=cost)

    rets = pd.read_parquet(RES / "test_returns.parquet")
    rf = rets.pop("_rf").to_numpy()
    if not rets.index.equals(res.dates):
        raise SystemExit("path mismatch with results/test_returns.parquet")
    chk = run(data, rules.fixed(0.7, cfg["baselines"]["fixed_exposure_horizon"]), start, end, cost_bps=cost)
    if not np.array_equal(chk.returns, rets["rp_fixed_0.7"].to_numpy()):
        raise SystemExit("engine rerun does not reproduce the protocol run's rp_fixed_0.7 returns")
    rets["vix_rule"] = res.returns
    dates = rets.index
    yr = dates.year

    extras = json.loads((RES / "extras.json").read_text())
    extras["vix_rule"] = {"costs": res.costs, "turnover": res.turnover,
                          "avg_exposure": float(res.decisions["exposure"].mean())}
    combined = pd.DataFrame({c: M.summary(rets[c].to_numpy(), rf, costs=extras[c]["costs"],
                                          turnover=extras[c]["turnover"], avg_exposure=extras[c]["avg_exposure"])
                             for c in NAMES}).T

    periods = {"2019-2026 (all)": np.ones(len(rets), bool), "2019-2021 (not previously seen)": yr <= 2021,
               "2022-2026 (previously seen)": yr >= 2022, "All years except 2020": yr != 2020}
    rows = []
    for pname, m in periods.items():
        for a, c in [("ppo_ensemble", "vix_rule"), ("vix_rule", "vol_target_floored")]:
            d = paired_sharpe_diff(rets.loc[m, a], rets.loc[m, c], rf[m], block=b["block"], draws=b["draws"],
                                   seed=b["seed"])
            rows.append({"period": pname, "comparison": f"{NAMES[a]} - {NAMES[c]}",
                         "sharpe_a": M.sharpe(rets.loc[m, a].to_numpy(), rf[m]),
                         "sharpe_b": M.sharpe(rets.loc[m, c].to_numpy(), rf[m]), "delta": d["observed"],
                         "p05": d["p05"], "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1],
                         "share_le_0": d["share_le_0"]})
    splits = pd.DataFrame(rows)

    per_year = pd.DataFrame({NAMES[c]: M.by_year(dates, rets[c].to_numpy(), rf)["sharpe"] for c in NAMES})
    per_year.index.name = "year"

    e_ppo = session_exposure(pd.read_csv(RES / "ensemble_decisions.csv", parse_dates=["date"]), dates)
    e_vix = session_exposure(res.decisions, dates)
    corr = float(np.corrcoef(e_ppo, e_vix)[0, 1])
    agree = float(((e_ppo >= 0.75) == (e_vix >= 0.75)).mean())

    tc = ledger.trial_counts()
    ex = rets["ppo_ensemble"].to_numpy() - rf
    d_this, d_prior = dsr(ex, max(1, tc["this_study"])), dsr(ex, max(1, tc["with_prior"]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selections.json").write_text(json.dumps(
        {str(y): {"params": s["params"], "val_sharpe": s["val_sharpe"],
                  "grid": [[g[0], g[1]] for g in s["grid"]]} for y, s in sel.items()}, indent=1, default=float))
    pd.DataFrame({"vix_rule": res.returns}, index=dates).to_parquet(OUT / "returns.parquet")
    res.decisions.to_csv(OUT / "decisions.csv", index=False)
    combined.rename_axis("strategy").to_csv(OUT / "combined.csv")
    splits.to_csv(OUT / "splits.csv", index=False)
    per_year.to_csv(OUT / "per_year_sharpe.csv")
    summary = {"exposure_corr_with_ppo": corr, "share_sessions_same_side_of_0.75": agree,
               "trial_counts_now": tc, "ppo_dsr_this_study_now": d_this["dsr"],
               "ppo_dsr_with_prior_now": d_prior["dsr"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    f3 = lambda x: f"{x:+.3f}"  # noqa: E731
    sel_rows = pd.DataFrame([{"fold": y, "threshold": f"{s['params']['threshold'] * 100:.0f}",
                              "horizon": s["params"]["horizon"], "val_sharpe": s["val_sharpe"]}
                             for y, s in sel.items()])
    comb = combined.reset_index().rename(columns={"index": "strategy"})
    comb["strategy"] = comb["strategy"].map(NAMES)
    lines = ["# VIX threshold rule (post-hoc, exploratory)", "",
             "Not pre-registered. Designed after the test run showed PPO's exposure rising with India VIX, and "
             "fixed in `eval/vix_rule.py` before it was run: exposure 1.0 when India VIX at the decision close is at "
             "or above a threshold, else 0.5; threshold {15, 20, 25} x horizon {5, 21} chosen per fold on the two "
             "validation years. Logged in ledger.csv before evaluation and counted as a trial.", "",
             "## Selected per fold", ""]
    lines += _table(sel_rows, ["fold", "threshold", "horizon", "val_sharpe"], {"val_sharpe": "{:.3f}".format})
    lines += ["", "## Combined path 2019-01-01..2026-11-06", ""]
    lines += _table(comb, ["strategy", "sharpe", "cagr", "vol", "max_drawdown", "avg_exposure"],
                    {"sharpe": "{:.3f}".format, "cagr": "{:.2%}".format, "vol": "{:.2%}".format,
                     "max_drawdown": "{:.2%}".format, "avg_exposure": "{:.2f}".format})
    lines += ["", "## Paired block bootstrap (same settings as H1)", ""]
    sp = splits.assign(ci=[f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(splits.ci_lo, splits.ci_hi)])
    lines += _table(sp, ["period", "comparison", "sharpe_a", "sharpe_b", "delta", "p05", "ci", "share_le_0"],
                    {"sharpe_a": "{:.3f}".format, "sharpe_b": "{:.3f}".format, "delta": f3, "p05": f3,
                     "share_le_0": "{:.3f}".format})
    lines += ["", "## Sharpe by year", ""]
    py = per_year.reset_index()
    lines += _table(py, list(py.columns), {c: "{:.3f}".format for c in py.columns if c != "year"})
    lines += ["", "## Agreement with PPO",
              "", f"Session-level exposure correlation with the PPO ensemble: {corr:+.2f}. Share of sessions on the "
              f"same side of 0.75 exposure: {agree:.2f}.", "",
              "## Deflated Sharpe of the PPO ensemble with the ledger as it now stands", "",
              f"This study's trials: {tc['this_study']} (DSR {d_this['dsr']:.3f}); with recorded prior work: "
              f"{tc['with_prior']} (DSR {d_prior['dsr']:.3f}). The protocol report computed these at the test run "
              "with one trial; quote the counts as of submission."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
