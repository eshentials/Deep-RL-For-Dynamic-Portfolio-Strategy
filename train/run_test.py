"""The single, gated test run (protocol Sections 6-7).

Protocol mode refuses to start unless tag protocol-v1 exists, protocol.yaml is
unchanged since it, the tree is clean and the risk-free series is the RBI one.
Every run is written to ledger.csv, so a rerun cannot happen silently.

--dev-dry-run exercises the whole path on development data only: the fold-2019 dev
models are evaluated on 2017-2018 (their validation years) as if they were test years,
and nothing is written to results/. It exists to test code, not to produce findings.

Run:  python -m train.run_test --protocol
      python -m train.run_test --dev-dry-run runs/dev/<stamp>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import rules
from baselines.select import select_best_fixed, select_vix_rule, select_vol_target
from config import ROOT, describe, load, protocol_tag, study_root
from env.build import equal_weights
from env.engine import run
from train.common import RUNS, Ensemble, FoldSwitch, PPOPolicy, folds, protocol_gate, run_tag


def load_models(run_dir: Path, fold_years, seeds):
    from stable_baselines3 import PPO
    out = {}
    for y in fold_years:
        for s in seeds:
            p = run_dir / f"fold_{y}" / f"seed_{s}" / "best.zip"
            if not p.exists():
                sys.exit(f"missing model {p}")
            out[(y, s)] = PPO.load(p, device="cpu")
    return out


def evaluate(data, fold_list, models, seeds, start, end, year_map, bench_tr):
    """Run every strategy on the path start..end. year_map: governed year -> fold year."""
    cfg = load()
    pf, bl = cfg["portfolio"], cfg["baselines"]
    cost = pf["cost_bps"]
    fold_by_year = {f.test_year: f for f in fold_list}

    def per_year_policy(build):
        return FoldSwitch(data, {y: build(fold_by_year[fy]) for y, fy in year_map.items()})

    strategies, extras = {}, {}

    def add(name, policy, **kw):
        res = run(data, policy, start, end, **kw)
        strategies[name] = res.returns
        extras[name] = {"costs": res.costs, "turnover": res.turnover,
                        "avg_exposure": float(res.decisions["exposure"].mean()) if len(res.decisions) else None}
        return res

    ens = lambda f: Ensemble([PPOPolicy(models[(f.test_year, s)], data, f) for s in seeds])  # noqa: E731
    ens_res = add("ppo_ensemble", per_year_policy(ens), cost_bps=cost)
    for s in seeds:
        add(f"ppo_seed_{s}", per_year_policy(lambda f, s=s: PPOPolicy(models[(f.test_year, s)], data, f)),
            cost_bps=cost)

    selections = {}
    for fy in sorted(set(year_map.values())):
        f = fold_by_year[fy]
        selections[fy] = {"vol_target_floored": select_vol_target(data, f, bl["vol_target"]["floor"]),
                          "vol_target_unfloored": select_vol_target(data, f, 0.0),
                          "best_fixed": select_best_fixed(data, f)}
        if "vix_rule" in bl:  # protocol v2: registered comparator for H2
            selections[fy]["vix_rule"] = select_vix_rule(data, f)

    def vt(kind):
        return lambda f: rules.vol_target(data, **_vt_kwargs(selections[f.test_year][kind]["params"]))
    add("vol_target_floored", per_year_policy(vt("vol_target_floored")), cost_bps=cost)
    add("vol_target_unfloored", per_year_policy(vt("vol_target_unfloored")), cost_bps=cost)
    def vix(f):
        return rules.vix_threshold(data, **selections[f.test_year]["vix_rule"]["params"])
    if "vix_rule" in bl:
        add("vix_rule", per_year_policy(vix), cost_bps=cost)
    add("best_fixed", per_year_policy(
        lambda f: rules.fixed(**_fx(selections[f.test_year]["best_fixed"]["params"]))), cost_bps=cost)
    for e in bl["fixed_exposures"]:
        add(f"rp_fixed_{e:g}", rules.fixed(e, bl["fixed_exposure_horizon"]), cost_bps=cost)
    add("equal_weight", rules.month_end_horizon(data), cost_bps=cost, weights=equal_weights(data.eligible))
    bk = cfg["data"].get("benchmark_key", "niftybees")
    strategies[bk] = rules.benchmark_buy_and_hold(bench_tr, data, start, end, cost)
    extras[bk] = {"costs": cost / 1e4, "turnover": 1.0, "avg_exposure": 1.0}
    for k in range(bl["random_policy"]["n_seeds"]):
        add(f"random_{k}", rules.random_policy(pf["exposure_grid"], pf["horizon_grid"], seed=k), cost_bps=cost)

    # Pre-registered sensitivities.
    add("ppo_ensemble_lag1", per_year_policy(ens), cost_bps=cost, lag=1)
    add("vol_target_floored_lag1", per_year_policy(vt("vol_target_floored")), cost_bps=cost, lag=1)
    if "vix_rule" in bl:
        add("vix_rule_lag1", per_year_policy(vix), cost_bps=cost, lag=1)
    add("ppo_ensemble_cost20", per_year_policy(ens), cost_bps=20)
    add("vol_target_floored_cost20", per_year_policy(vt("vol_target_floored")), cost_bps=20)
    if "vix_rule" in bl:
        add("vix_rule_cost20", per_year_policy(vix), cost_bps=20)
    add("equal_weight_cost20", rules.month_end_horizon(data), cost_bps=20, weights=equal_weights(data.eligible))
    n_delist = int(data.delist_mask[start:end + 1].sum())
    extras["_delisting_haircut"] = {"applicable": n_delist > 0, "delisting_exits_on_path": n_delist}
    if n_delist:
        add("ppo_ensemble_haircut", per_year_policy(ens), cost_bps=cost,
            haircut=cfg["data"]["delisting_haircut_sensitivity"])

    dates = data.dates[start + 1:end + 1]
    rets = pd.DataFrame(strategies, index=dates)
    rets["_rf"] = data.rf[start + 1:end + 1]
    return rets, extras, selections, ens_res.decisions


def _vt_kwargs(p):
    return {"target": p["target"], "horizon": p["horizon"], "floor": p["floor"], "cap": p["cap"],
            "window": p["window"]}


def _fx(p):
    return {"e": p["e"], "h": p["h"]}


def _jsonable(sel):
    return {str(k): {n: {"params": v["params"], "val_sharpe": v["val_sharpe"],
                         "grid": [[g[0], g[1]] for g in v["grid"]]} for n, v in d.items()}
            for k, d in sel.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--protocol", action="store_true")
    g.add_argument("--dev-dry-run", metavar="DEV_RUN_DIR")
    a = ap.parse_args(argv)
    cfg = load()
    from data.panel import load_engine
    data = load_engine()
    bench = pd.read_parquet(study_root() / "data" / "processed" / "market.parquet")["bench_tr"]
    print(f"protocol {describe()}")
    seeds = cfg["training"]["seeds"]

    if a.protocol:
        problems = protocol_gate()
        if problems:
            sys.exit("refusing the test run:\n  - " + "\n  - ".join(problems))
        years = cfg["folds"]["test_years"]
        fl = folds(data, years)
        models = load_models(RUNS / "protocol", years, seeds)
        start = data.last_session_before(min(years))
        end = data.last_session_of(max(years))
        year_map = {y: y for y in years}
        out = study_root() / "results"
        # Record the attempt before any test-year score exists, so a crash and rerun
        # cannot go unrecorded (the gate then needs this row committed to rerun).
        from train import ledger
        what = "" if cfg["version"] == 1 else f" ({cfg['data']['universe']})"
        ledger.append("test_run", f"{protocol_tag()} test run{what}: PPO ensemble, seeds and all baselines on 2019..freeze",
                      folds=" ".join(map(str, years)), seeds=" ".join(map(str, seeds)),
                      touched_test=True, counts_as_trial=False,
                      notes=f"{out.relative_to(ROOT)}/; {run_tag()}; logged at start, before evaluation")
    else:
        run_dir = ROOT / a.dev_dry_run
        fl = folds(data, [2019])
        have = sorted(int(p.name.split("_")[1]) for p in (run_dir / "fold_2019").glob("seed_*"))
        seeds = have
        models = load_models(run_dir, [2019], seeds)
        start, end = data.last_session_before(2017), data.last_session_of(2018)
        year_map = {2017: 2019, 2018: 2019}
        out = run_dir / "dry_run"

    rets, extras, selections, decisions = evaluate(data, fl, models, seeds, start, end, year_map, bench)
    out.mkdir(parents=True, exist_ok=True)
    rets.to_parquet(out / "test_returns.parquet")
    decisions.to_csv(out / "ensemble_decisions.csv", index=False)
    (out / "extras.json").write_text(json.dumps(extras, indent=1, default=float))
    (out / "selections.json").write_text(json.dumps(_jsonable(selections), indent=1, default=float))
    (out / "run_info.json").write_text(json.dumps({"run_tag": run_tag(), "seeds": seeds,
                                                   "mode": "protocol" if a.protocol else "dev-dry-run",
                                                   "start": str(data.dates[start].date()),
                                                   "end": str(data.dates[end].date())}, indent=1))
    from eval.report import report
    report(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
