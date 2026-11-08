"""Generate docs/methods_facts.md: exact numbers for the methods section.

Built only from config/protocol.yaml and the data-build reports, never from test-year
results, so it is safe to circulate before the protocol is tagged.

Run:  python -m eval.methods_facts
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import ROOT, load, protocol_hash, require_version
from data.rf import is_provisional

PROC, ROSTER, DOCS = ROOT / "data" / "processed", ROOT / "data" / "roster", ROOT / "docs"


def build() -> str:
    require_version(1, "eval.methods_facts")
    cfg = load()
    iv = pd.read_csv(ROSTER / "membership_intervals.csv")
    ch = pd.read_csv(ROSTER / "changes.csv")
    val = pd.read_csv(ROSTER / "validation.csv")
    idx = pd.read_csv(ROSTER / "snapshot_index.csv")
    qa = pd.read_csv(PROC / "price_qa.csv")
    rep = pd.read_csv(PROC / "price_repairs.csv")
    brep = pd.read_csv(PROC / "benchmark_repairs.csv")
    bdev = pd.read_csv(PROC / "benchmark_tracking_deviations.csv")
    z = np.load(PROC / "engine.npz")
    dates = pd.DatetimeIndex(z["dates"].astype("datetime64[ns]"))
    elig = pd.Series(z["eligible"].sum(axis=1), index=dates)["2014":]
    excluded = qa.loc[qa["status"] == "excluded", "symbol"].tolist()
    pf, tr, bl, ev = cfg["portfolio"], cfg["training"], cfg["baselines"], cfg["evaluation"]
    n_changes = int(ch["action"].isin(["add", "remove"]).sum())
    n_dates = ch.loc[ch["action"].isin(["add", "remove"]), "effective_date"].nunique()

    L = ["# Methods facts", "",
         f"Generated from `config/protocol.yaml` (SHA-256 `{protocol_hash()[:16]}...`) and the data-build reports. "
         "No test-year results are used. Regenerate with `python -m eval.methods_facts`.", "",
         "## Universe and data", "",
         f"- Point-in-time Nifty 50 membership, rebuilt from {len(idx)} archived captures of NSE's constituent "
         f"files and API ({idx['members_hash'].nunique()} distinct member sets). {n_changes} add/remove events on "
         f"{n_dates} effective dates, each dated from an NSE notice or a news report (`data/roster/changes.csv`).",
         f"- Validation: the rebuilt roster matches {int(val['match'].sum())} of {len(val)} snapshots taken on or after "
         "2014-01-01. The one mismatch is a 2015-05-19 copy of the pre-rename file that NSE had not yet updated. "
         "There are exactly 50 members on every session, or 51 while Tata Motors DVR was an additional security "
         "(2016-04-01 to 2017-09-28).",
         f"- {iv['symbol'].nunique()} symbols were members between {cfg['data']['study_start']} and the data freeze "
         f"({dates[-1].date()}). {int((qa['status'] == 'adjusted').sum())} have usable Yahoo Finance adjusted prices. "
         f"{len(excluded)} have no usable series and are excluded during their membership: {', '.join(excluded)}.",
         f"- Eligible stocks per session, 2014 onward: min {int(elig.min())}, median {int(elig.median())}, "
         f"max {int(elig.max())}.",
         f"- Corporate-action repairs: {len(rep)} in all, each logged with its source.",
         ]
    for r in rep.itertuples():
        L.append(f"  - {r.symbol} {r.date}: {r.kind.replace('_', ' ')}, {r.detail}.")
    L += [f"- Benchmark (NIFTYBEES adjusted close): {len(brep)} sessions repaired (vendor level glitch or missing). "
          f"{len(bdev)} genuine tracking gaps are kept and logged.",
          f"- Risk-free rate: {cfg['risk_free']['series']}, carried forward, accrued over calendar days / "
          f"{cfg['risk_free']['day_count']}. Status: "
          f"{'PROVISIONAL placeholder, RBI file not yet added' if is_provisional() else 'RBI series loaded'}.",
          "", "## Portfolio, costs and execution", "",
          f"- Equal risk contribution over eligible stocks, Ledoit-Wolf covariance of the trailing {pf['cov_window']} sessions.",
          f"- Stock book = exposure x risk-parity weights. Cash (1 - exposure) earns the risk-free rate.",
          f"- Costs: {pf['cost_bps']} bp per unit of one-way turnover. Decide and trade at close t, earn from t+1.",
          "- A decision is forced at each year-end close, made by the next year's fold policy.",
          "", "## Agent", "",
          f"- Action: exposure in {pf['exposure_grid']}, horizon in {pf['horizon_grid']} sessions (MultiDiscrete([4, 2])).",
          f"- State ({len(cfg['state']['market_features']) + len(cfg['state']['path_features'])} features): "
          f"{', '.join(cfg['state']['market_features'] + cfg['state']['path_features'])}.",
          f"- Reward per interval: sum of daily excess log returns minus kappa x drawdown increase, kappa = "
          f"{cfg['reward']['kappa']}, scaled x{cfg['reward']['scale']:g}.",
          f"- PPO (Stable-Baselines3): {json.dumps(tr['ppo'])}.",
          f"- {len(tr['seeds'])} seeds x {tr['total_steps']:,} steps per fold. Checkpoints every {tr['eval_every']:,} "
          f"steps are scored by the mean Sharpe over the two validation years. Episodes are {tr['episode_sessions']} "
          "sessions, starting in cash.",
          "- Ensemble: mean exposure and median horizon of the 5 seeds, queried on the ensemble's own state.",
          "", "## Walk-forward folds", "", "| Test year | Train | Validate |", "|---|---|---|"]
    for Y in cfg["folds"]["test_years"]:
        L.append(f"| {Y}{' (previously seen)' if Y in cfg['folds']['previously_seen'] else ''} | "
                 f"{cfg['folds']['train_start_year']}-{Y - 3} | {Y - 2}-{Y - 1} |")
    vt = bl["vol_target"]
    L += ["", "## Baselines (same engine, costs and timing)", "",
          "- NIFTYBEES buy-and-hold. Equal weight, rebalanced at month-end.",
          f"- Risk parity at fixed exposure {bl['fixed_exposures']}, {bl['fixed_exposure_horizon']}-session rebalance.",
          f"- Vol targeting: e = clip(target / realised vol of the risk-parity book over {vt['vol_window']} sessions, "
          f"{vt['floor']}, {vt['cap']}). Target in {vt['targets']} and horizon in {vt['horizons']}, chosen per fold "
          "on validation. The unfloored rule is secondary.",
          f"- Random policy over the same 8 actions ({bl['random_policy']['n_seeds']} seeds). "
          "Best fixed action, chosen per fold on validation.",
          "", "## Statistics", "",
          f"- H1: one-sided paired circular block bootstrap of the Sharpe difference (PPO ensemble minus floored vol "
          f"targeting), block {ev['bootstrap']['block']}, {ev['bootstrap']['draws']:,} draws, seed "
          f"{ev['bootstrap']['seed']}. Supported if the 5th percentile is above 0.",
          "- Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014), with the null Sharpe variance 1/(T-1). Trial "
          "counts are taken from `ledger.csv`, reported for this study alone and with recorded prior work.",
          f"- Pre-registered sensitivities: {', '.join(ev['sensitivities'])}.",
          "", "## Compute", "",
          "- Apple M2 (8 cores, 8 GB), CPU only, one PyTorch thread per worker, 5 workers.",
          "- A development run of fold 2019 (5 seeds x 200k steps) took 2.2 minutes.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    (DOCS / "methods_facts.md").write_text(build())
    print(f"wrote {DOCS / 'methods_facts.md'}")
