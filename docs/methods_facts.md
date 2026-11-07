# Methods facts

Generated from `config/protocol.yaml` (SHA-256 `c97977a4a4bbb055...`) and the data-build reports. No test-year results are used. Regenerate with `python -m eval.methods_facts`.

## Universe and data

- Point-in-time Nifty 50 membership, rebuilt from 76 archived captures of NSE's constituent files and API (19 distinct member sets). 78 add/remove events on 25 effective dates, each dated from an NSE notice or a news report (`data/roster/changes.csv`).
- Validation: the rebuilt roster matches 67 of 68 snapshots taken on or after 2014-01-01. The one mismatch is a 2015-05-19 copy of the pre-rename file that NSE had not yet updated. There are exactly 50 members on every session, or 51 while Tata Motors DVR was an additional security (2016-04-01 to 2017-09-28).
- 87 symbols were members between 2014-01-01 and the data freeze (2026-11-06). 81 have usable Yahoo Finance adjusted prices. 6 have no usable series and are excluded during their membership: CAIRN, HDFC, IDFC, JPASSOCIAT, RANBAXY, TATAMTRDVR.
- Eligible stocks per session, 2014 onward: min 45, median 49, max 50.
- Corporate-action repairs: 4 in all, each logged with its source.
  - ITC 2025-01-06: demerger, discovered 455.6 vs prev close 481.6; ex-date return -4.55% -> -2.84%.
  - RELIANCE 2023-07-20: demerger, discovered 2580.0 vs prev close 2841.85; ex-date return -0.12% -> +1.54%.
  - TMPV 2025-10-14: demerger, discovered 400.0 vs prev close 660.75; ex-date return -40.15% -> -1.14%.
  - TRENT 2026-01-01: split boundary artifact, split 1.5 on 2026-06-04; raw return -33.05% -> +0.43%.
- Benchmark (NIFTYBEES adjusted close): 3 sessions repaired (vendor level glitch or missing). 3 genuine tracking gaps are kept and logged.
- Risk-free rate: RBI 91-day Treasury bill primary auction cut-off yield (weekly), carried forward, accrued over calendar days / 365. Status: RBI series loaded.

## Portfolio, costs and execution

- Equal risk contribution over eligible stocks, Ledoit-Wolf covariance of the trailing 120 sessions.
- Stock book = exposure x risk-parity weights. Cash (1 - exposure) earns the risk-free rate.
- Costs: 10 bp per unit of one-way turnover. Decide and trade at close t, earn from t+1.
- A decision is forced at each year-end close, made by the next year's fold policy.

## Agent

- Action: exposure in [0.5, 0.7, 0.85, 1.0], horizon in [5, 21] sessions (MultiDiscrete([4, 2])).
- State (13 features): idx_vol20, idx_vol60, idx_vol_ratio, idx_mom21, idx_mom63, idx_dd252, avg_corr60, vix, vix_chg5, fx_chg21_lag1, exposure, own_dd, own_vol20.
- Reward per interval: sum of daily excess log returns minus kappa x drawdown increase, kappa = 1.0, scaled x100.
- PPO (Stable-Baselines3): {"learning_rate": 0.0003, "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 1.0, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "net_arch": [64, 64]}.
- 5 seeds x 200,000 steps per fold. Checkpoints every 10,000 steps are scored by the mean Sharpe over the two validation years. Episodes are 252 sessions, starting in cash.
- Ensemble: mean exposure and median horizon of the 5 seeds, queried on the ensemble's own state.

## Walk-forward folds

| Test year | Train | Validate |
|---|---|---|
| 2019 | 2014-2016 | 2017-2018 |
| 2020 | 2014-2017 | 2018-2019 |
| 2021 | 2014-2018 | 2019-2020 |
| 2022 (previously seen) | 2014-2019 | 2020-2021 |
| 2023 (previously seen) | 2014-2020 | 2021-2022 |
| 2024 (previously seen) | 2014-2021 | 2022-2023 |
| 2025 (previously seen) | 2014-2022 | 2023-2024 |
| 2026 (previously seen) | 2014-2023 | 2024-2025 |

## Baselines (same engine, costs and timing)

- NIFTYBEES buy-and-hold. Equal weight, rebalanced at month-end.
- Risk parity at fixed exposure [0.5, 0.7, 0.85, 1.0], 21-session rebalance.
- Vol targeting: e = clip(target / realised vol of the risk-parity book over 60 sessions, 0.5, 1.0). Target in [0.1, 0.12, 0.15] and horizon in [5, 21], chosen per fold on validation. The unfloored rule is secondary.
- Random policy over the same 8 actions (100 seeds). Best fixed action, chosen per fold on validation.

## Statistics

- H1: one-sided paired circular block bootstrap of the Sharpe difference (PPO ensemble minus floored vol targeting), block 20, 10,000 draws, seed 20261107. Supported if the 5th percentile is above 0.
- Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014), with the null Sharpe variance 1/(T-1). Trial counts are taken from `ledger.csv`, reported for this study alone and with recorded prior work.
- Pre-registered sensitivities: execution_lag_1, cost_20bps, vol_target_unfloored, delisting_haircut.

## Compute

- Apple M2 (8 cores, 8 GB), CPU only, one PyTorch thread per worker, 5 workers.
- A development run of fold 2019 (5 seeds x 200k steps) took 2.2 minutes.
