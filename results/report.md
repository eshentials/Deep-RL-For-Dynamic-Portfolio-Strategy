# Test report (protocol)

Path: 2018-12-31 .. 2026-11-06 (1936 sessions).

## Primary hypothesis (H1)

Sharpe(PPO ensemble) - Sharpe(vol targeting, floor 0.5) = **+0.320**; bootstrap 5th percentile +0.054; 95% CI [+0.000, +0.620]; share of draws <= 0: 0.025.

**H1 SUPPORTED** under the pre-registered rule (5th percentile above zero).

## Combined path

| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Costs | Avg exposure |
|---|---|---|---|---|---|---|---|
| PPO ensemble | 0.808 | 14.91% | 11.42% | -20.67% | 0.721 | 2.65% | 0.66 |
| Vol targeting (floor 0.5) | 0.488 | 11.37% | 12.89% | -29.89% | 0.381 | 2.73% | 0.87 |
| Vol targeting (no floor) | 0.483 | 11.54% | 13.51% | -29.89% | 0.386 | 2.77% | 0.93 |
| Best fixed action | 0.547 | 12.21% | 12.87% | -34.87% | 0.350 | 2.04% | 0.65 |
| Equal weight (monthly) | 0.662 | 16.42% | 17.20% | -38.26% | 0.429 | 1.21% | 1.00 |
| NIFTYBEES buy-and-hold | 0.503 | 12.65% | 15.53% | -36.34% | 0.348 | 0.10% | 1.00 |
| Risk parity, e=0.50 | 0.624 | 10.37% | 7.79% | -17.41% | 0.596 | 0.82% | 0.50 |
| Risk parity, e=0.70 | 0.619 | 12.18% | 10.96% | -24.33% | 0.501 | 1.24% | 0.70 |
| Risk parity, e=0.85 | 0.615 | 13.48% | 13.37% | -29.69% | 0.454 | 1.59% | 0.85 |
| Risk parity, e=1.00 | 0.611 | 14.71% | 15.81% | -35.03% | 0.420 | 1.98% | 1.00 |
| ppo_seed_0 | 0.809 | 14.88% | 11.38% | -18.41% | 0.809 | 3.07% | 0.64 |
| ppo_seed_1 | 0.681 | 12.97% | 10.99% | -18.06% | 0.718 | 3.18% | 0.65 |
| ppo_seed_2 | 0.876 | 15.87% | 11.50% | -18.08% | 0.878 | 3.02% | 0.63 |
| ppo_seed_3 | 0.810 | 15.03% | 11.56% | -17.45% | 0.862 | 3.43% | 0.65 |
| ppo_seed_4 | 0.611 | 13.11% | 12.82% | -30.87% | 0.425 | 2.60% | 0.65 |
| ppo_ensemble_lag1 | 0.808 | 15.01% | 11.56% | -22.80% | 0.659 | 2.68% | 0.67 |
| vol_target_floored_lag1 | 0.532 | 12.02% | 12.93% | -28.63% | 0.420 | 2.83% | 0.87 |
| ppo_ensemble_cost20 | 0.791 | 14.68% | 11.42% | -20.86% | 0.704 | 5.24% | 0.67 |
| vol_target_floored_cost20 | 0.470 | 11.11% | 12.89% | -30.06% | 0.370 | 5.40% | 0.87 |
| equal_weight_cost20 | 0.656 | 16.32% | 17.20% | -38.30% | 0.426 | 2.41% | 1.00 |
| random (median of 100) | 0.567 |  |  |  |  |  |  |

## Deflated Sharpe Ratio (PPO ensemble)

- This study's trials (N=1): DSR 0.987, PSR 0.987
- With recorded prior work (N=83): DSR 0.412

## Secondary outcomes (exploratory intervals)

| vs | Metric | PPO - baseline | 95% CI |
|---|---|---|---|
| Equal weight (monthly) | max_drawdown | +0.176 | [+0.041, +0.249] |
| Equal weight (monthly) | calmar | +0.292 | [+0.050, +0.808] |
| NIFTYBEES buy-and-hold | max_drawdown | +0.157 | [+0.032, +0.278] |
| NIFTYBEES buy-and-hold | calmar | +0.373 | [+0.098, +1.151] |

## Sharpe differences vs other baselines (exploratory)

| vs | PPO - baseline | 95% CI |
|---|---|---|
| Vol targeting (floor 0.5) | +0.320 | [+0.002, +0.619] |
| Vol targeting (no floor) | +0.326 | [+0.034, +0.636] |
| Best fixed action | +0.261 | [-0.149, +0.550] |
| Equal weight (monthly) | +0.147 | [-0.078, +0.345] |
| NIFTYBEES buy-and-hold | +0.305 | [-0.041, +0.608] |
| Risk parity, e=0.50 | +0.185 | [-0.016, +0.374] |
| Risk parity, e=0.70 | +0.190 | [-0.015, +0.379] |
| Risk parity, e=0.85 | +0.193 | [-0.016, +0.385] |
| Risk parity, e=1.00 | +0.197 | [-0.015, +0.391] |
| ppo_seed_0 | -0.000 | [-0.129, +0.133] |
| ppo_seed_1 | +0.127 | [-0.012, +0.303] |
| ppo_seed_2 | -0.068 | [-0.191, +0.063] |
| ppo_seed_3 | -0.001 | [-0.165, +0.174] |
| ppo_seed_4 | +0.198 | [-0.058, +0.397] |

Random policy Sharpe: median 0.567, 5-95% band [0.410, 0.759].

## Per year (previously seen years marked *)

| Year | PPO ens. | Vol tgt | Best fixed | Equal wt | NIFTYBEES | PPO - VT [95% CI] |
|---|---|---|---|---|---|---|
| 2019 | 0.077 | 0.125 | 0.139 | 0.037 | 0.618 | -0.048 [-0.593, +0.489] |
| 2020 | 1.491 | 0.317 | 0.707 | 0.699 | 0.533 | +1.175 [-0.117, +1.847] |
| 2021 | 1.630 | 1.666 | 1.780 | 1.925 | 1.473 | -0.036 [-0.949, +0.690] |
| 2022* | 0.308 | 0.025 | 0.145 | 0.247 | 0.106 | +0.283 [-0.086, +0.703] |
| 2023* | 2.400 | 2.380 | 2.376 | 2.232 | 1.452 | +0.020 [-0.004, +0.042] |
| 2024* | 0.539 | 0.259 | 0.287 | 0.328 | 0.344 | +0.280 [-0.036, +0.628] |
| 2025* | 0.483 | 0.502 | 0.532 | 0.693 | 0.544 | -0.019 [-0.231, +0.170] |
| 2026* | -0.379 | -0.573 | -0.537 | -0.136 | -0.624 | +0.194 [-0.341, +0.665] |

## Drawdown episodes (NIFTYBEES's deepest episode with its peak in the year)

| Year | Peak | Trough | Strategy | Peak-to-trough | Max DD in window |
|---|---|---|---|---|---|
| 2020 | 2020-01-17 | 2020-03-23 | PPO ensemble | -20.04% | -20.04% |
| 2020 | 2020-01-17 | 2020-03-23 | Vol targeting (floor 0.5) | -29.09% | -29.09% |
| 2020 | 2020-01-17 | 2020-03-23 | Vol targeting (no floor) | -29.09% | -29.09% |
| 2020 | 2020-01-17 | 2020-03-23 | Best fixed action | -34.87% | -34.87% |
| 2020 | 2020-01-17 | 2020-03-23 | Equal weight (monthly) | -36.70% | -36.70% |
| 2020 | 2020-01-17 | 2020-03-23 | NIFTYBEES buy-and-hold | -36.34% | -36.34% |
| 2025 | 2025-10-29 | 2025-11-06 | PPO ensemble | -1.08% | -1.08% |
| 2025 | 2025-10-29 | 2025-11-06 | Vol targeting (floor 0.5) | -2.23% | -2.23% |
| 2025 | 2025-10-29 | 2025-11-06 | Vol targeting (no floor) | -2.23% | -2.23% |
| 2025 | 2025-10-29 | 2025-11-06 | Best fixed action | -1.06% | -1.06% |
| 2025 | 2025-10-29 | 2025-11-06 | Equal weight (monthly) | -2.30% | -2.30% |
| 2025 | 2025-10-29 | 2025-11-06 | NIFTYBEES buy-and-hold | -2.13% | -2.13% |

Delisting haircut sensitivity: not applicable (0 delisting exits on the path).

Figures: figures/equity, drawdown, exposure_vs_vix, per_year_delta_sharpe, seed_spread (png and pdf).
