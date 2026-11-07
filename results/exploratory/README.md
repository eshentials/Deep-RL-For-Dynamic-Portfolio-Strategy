# Exploratory analyses (post-unblinding, not pre-registered)

These re-read the protocol test run's returns. They are not part of H1 and do not change its verdict. Logged in ledger.csv as kind `exploratory`.

## 1. Period splits (paired block bootstrap, same settings as H1)

| period | vs | sharpe_ppo | sharpe_other | delta | p05 | ci | share_le_0 |
|---|---|---|---|---|---|---|---|
| 2019-2021 (not previously seen) | Vol targeting (floor 0.5) | 1.162 | 0.555 | +0.607 | +0.114 | [+0.017, +1.055] | 0.022 |
| 2019-2021 (not previously seen) | Risk parity, e=0.70 (nearest fixed exposure to the ensemble's 0.66 average) | 1.162 | 0.825 | +0.337 | -0.001 | [-0.055, +0.633] | 0.051 |
| 2019-2021 (not previously seen) | Equal weight (monthly) | 1.162 | 0.799 | +0.363 | -0.008 | [-0.079, +0.686] | 0.053 |
| 2019-2021 (not previously seen) | NIFTYBEES buy-and-hold | 1.162 | 0.757 | +0.405 | -0.154 | [-0.258, +0.891] | 0.120 |
| 2022-2026 (previously seen) | Vol targeting (floor 0.5) | 0.487 | 0.443 | +0.045 | -0.260 | [-0.316, +0.413] | 0.409 |
| 2022-2026 (previously seen) | Risk parity, e=0.70 (nearest fixed exposure to the ensemble's 0.66 average) | 0.487 | 0.451 | +0.036 | -0.139 | [-0.171, +0.240] | 0.366 |
| 2022-2026 (previously seen) | Equal weight (monthly) | 0.487 | 0.561 | -0.074 | -0.253 | [-0.287, +0.137] | 0.757 |
| 2022-2026 (previously seen) | NIFTYBEES buy-and-hold | 0.487 | 0.285 | +0.202 | -0.084 | [-0.146, +0.545] | 0.121 |
| All years except 2020 | Vol targeting (floor 0.5) | 0.649 | 0.548 | +0.101 | -0.149 | [-0.194, +0.400] | 0.252 |
| All years except 2020 | Risk parity, e=0.70 (nearest fixed exposure to the ensemble's 0.66 average) | 0.649 | 0.611 | +0.038 | -0.114 | [-0.143, +0.221] | 0.343 |
| All years except 2020 | Equal weight (monthly) | 0.649 | 0.691 | -0.042 | -0.205 | [-0.240, +0.158] | 0.660 |
| All years except 2020 | NIFTYBEES buy-and-hold | 0.649 | 0.526 | +0.123 | -0.125 | [-0.173, +0.410] | 0.202 |

## 2. Deepest NIFTYBEES drawdown episodes on the whole path

| rank | peak | trough | strategy | return_peak_to_trough | max_dd_in_window |
|---|---|---|---|---|---|
| 1 | 2020-01-17 | 2020-03-23 | ppo_ensemble | -20.0% | -20.0% |
| 1 | 2020-01-17 | 2020-03-23 | vol_target_floored | -29.1% | -29.1% |
| 1 | 2020-01-17 | 2020-03-23 | rp_fixed_0.7 | -24.3% | -24.3% |
| 1 | 2020-01-17 | 2020-03-23 | equal_weight | -36.7% | -36.7% |
| 1 | 2020-01-17 | 2020-03-23 | niftybees | -36.3% | -36.3% |
| 2 | 2021-10-18 | 2022-06-17 | ppo_ensemble | -10.8% | -10.8% |
| 2 | 2021-10-18 | 2022-06-17 | vol_target_floored | -11.2% | -11.2% |
| 2 | 2021-10-18 | 2022-06-17 | rp_fixed_0.7 | -10.8% | -10.8% |
| 2 | 2021-10-18 | 2022-06-17 | equal_weight | -16.4% | -16.4% |
| 2 | 2021-10-18 | 2022-06-17 | niftybees | -16.1% | -16.1% |
| 3 | 2024-09-27 | 2025-03-04 | ppo_ensemble | -8.9% | -8.9% |
| 3 | 2024-09-27 | 2025-03-04 | vol_target_floored | -18.6% | -18.6% |
| 3 | 2024-09-27 | 2025-03-04 | rp_fixed_0.7 | -12.8% | -12.8% |
| 3 | 2024-09-27 | 2025-03-04 | equal_weight | -17.9% | -18.0% |
| 3 | 2024-09-27 | 2025-03-04 | niftybees | -15.2% | -15.2% |

## 3. Ensemble exposure by the market state at the decision

Correlation of chosen exposure with India VIX: +0.62; with the Nifty drawdown from its 252-session high: -0.29 (negative = more exposure in deeper drawdowns).

| vix_bucket | decisions | mean_exposure | share_at_floor |
|---|---|---|---|
| <15 | 51 | 0.54 | 0.71 |
| 15-20 | 38 | 0.67 | 0.32 |
| 20-25 | 20 | 0.86 | 0.05 |
| >=25 | 9 | 0.87 | 0.00 |

| dd_bucket | decisions | mean_exposure | share_at_floor |
|---|---|---|---|
| < -20% | 2 | 1.00 | 0.00 |
| -20% to -10% | 19 | 0.74 | 0.11 |
| -10% to -5% | 31 | 0.64 | 0.45 |
| -5% to 0% | 66 | 0.64 | 0.50 |
