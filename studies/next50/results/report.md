# Test report (protocol)

Path: 2018-12-31 .. 2026-11-06 (1936 sessions).

## Primary hypotheses (Holm, family-wise alpha 0.05)

p is the share of paired-bootstrap draws of the Sharpe difference at or below zero. Holm rejects the k-th smallest p while p <= alpha / (m - k + 1).

| Hypothesis | Comparison | Sharpe difference | 95% CI | p | Holm threshold | Decision |
|---|---|---|---|---|---|---|
| H1 | PPO ensemble - Vol targeting (floor 0.5) | **+0.195** | [-0.088, +0.447] | 0.0851 | 0.0250 | **NOT SUPPORTED** |
| H2 | PPO ensemble - VIX threshold rule | **-0.028** | [-0.325, +0.213] | 0.6132 | 0.0500 | **NOT SUPPORTED** |

## Combined path

| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Costs | Avg exposure |
|---|---|---|---|---|---|---|---|
| PPO ensemble | 0.639 | 13.81% | 13.34% | -27.48% | 0.503 | 2.85% | 0.73 |
| Vol targeting (floor 0.5) | 0.444 | 10.97% | 13.60% | -33.76% | 0.325 | 2.49% | 0.78 |
| Vol targeting (no floor) | 0.413 | 10.70% | 14.22% | -33.76% | 0.317 | 2.50% | 0.87 |
| VIX threshold rule | 0.667 | 13.62% | 12.32% | -30.05% | 0.453 | 3.01% | 0.61 |
| Best fixed action | 0.507 | 10.25% | 9.70% | -17.19% | 0.596 | 1.44% | 0.53 |
| Equal weight (monthly) | 0.552 | 14.90% | 18.71% | -36.98% | 0.403 | 1.79% | 1.00 |
| JUNIORBEES buy-and-hold | 0.514 | 13.59% | 17.45% | -34.51% | 0.394 | 0.10% | 1.00 |
| Risk parity, e=0.50 | 0.602 | 10.51% | 8.34% | -17.19% | 0.611 | 1.08% | 0.50 |
| Risk parity, e=0.70 | 0.600 | 12.37% | 11.73% | -23.99% | 0.515 | 1.61% | 0.70 |
| Risk parity, e=0.85 | 0.598 | 13.69% | 14.29% | -28.94% | 0.473 | 2.07% | 0.85 |
| Risk parity, e=1.00 | 0.596 | 14.96% | 16.87% | -33.76% | 0.443 | 2.56% | 1.00 |
| ppo_seed_0 | 0.630 | 14.00% | 13.92% | -30.05% | 0.466 | 3.27% | 0.74 |
| ppo_seed_1 | 0.658 | 13.99% | 13.14% | -30.10% | 0.465 | 3.70% | 0.66 |
| ppo_seed_2 | 0.770 | 14.61% | 11.70% | -17.19% | 0.850 | 3.90% | 0.69 |
| ppo_seed_3 | 0.559 | 13.06% | 14.28% | -30.02% | 0.435 | 3.25% | 0.76 |
| ppo_seed_4 | 0.576 | 13.47% | 14.58% | -30.05% | 0.448 | 3.58% | 0.77 |
| ppo_ensemble_lag1 | 0.626 | 13.60% | 13.30% | -27.21% | 0.500 | 2.84% | 0.73 |
| vol_target_floored_lag1 | 0.461 | 11.24% | 13.62% | -33.63% | 0.334 | 2.53% | 0.78 |
| vix_rule_lag1 | 0.662 | 13.50% | 12.26% | -29.71% | 0.454 | 3.03% | 0.61 |
| ppo_ensemble_cost20 | 0.622 | 13.55% | 13.33% | -27.51% | 0.493 | 5.64% | 0.73 |
| vol_target_floored_cost20 | 0.427 | 10.73% | 13.60% | -33.77% | 0.318 | 4.93% | 0.78 |
| vix_rule_cost20 | 0.649 | 13.36% | 12.32% | -30.09% | 0.444 | 5.96% | 0.61 |
| equal_weight_cost20 | 0.545 | 14.74% | 18.71% | -36.99% | 0.398 | 3.56% | 1.00 |
| random (median of 100) | 0.538 |  |  |  |  |  |  |

## Deflated Sharpe Ratio (PPO ensemble)

- This study's trials (N=3): DSR 0.813, PSR 0.957
- With recorded prior work (N=85): DSR 0.248

## Secondary outcomes (exploratory intervals)

| vs | Metric | PPO - baseline | 95% CI |
|---|---|---|---|
| Equal weight (monthly) | max_drawdown | +0.095 | [+0.043, +0.186] |
| Equal weight (monthly) | calmar | +0.100 | [-0.043, +0.539] |
| JUNIORBEES buy-and-hold | max_drawdown | +0.070 | [+0.039, +0.200] |
| JUNIORBEES buy-and-hold | calmar | +0.109 | [-0.022, +0.671] |

## Sharpe differences vs other baselines (exploratory)

| vs | PPO - baseline | 95% CI |
|---|---|---|
| Vol targeting (floor 0.5) | +0.195 | [-0.104, +0.449] |
| Vol targeting (no floor) | +0.226 | [-0.032, +0.475] |
| VIX threshold rule | -0.028 | [-0.328, +0.216] |
| Best fixed action | +0.132 | [-0.172, +0.469] |
| Equal weight (monthly) | +0.086 | [-0.091, +0.259] |
| JUNIORBEES buy-and-hold | +0.125 | [-0.093, +0.352] |
| Risk parity, e=0.50 | +0.036 | [-0.121, +0.213] |
| Risk parity, e=0.70 | +0.039 | [-0.117, +0.213] |
| Risk parity, e=0.85 | +0.041 | [-0.111, +0.215] |
| Risk parity, e=1.00 | +0.042 | [-0.108, +0.213] |
| ppo_seed_0 | +0.009 | [-0.119, +0.124] |
| ppo_seed_1 | -0.020 | [-0.214, +0.133] |
| ppo_seed_2 | -0.132 | [-0.359, +0.109] |
| ppo_seed_3 | +0.079 | [-0.026, +0.184] |
| ppo_seed_4 | +0.063 | [-0.042, +0.184] |

Random policy Sharpe: median 0.538, 5-95% band [0.383, 0.688].

## Per year (previously seen years marked *)

| Year | PPO ens. | Vol tgt | Best fixed | Equal wt | JUNIORBEES | PPO - VT [95% CI] |
|---|---|---|---|---|---|---|
| 2019* | 0.229 | -0.024 | 0.042 | 0.004 | -0.260 | +0.253 [+0.031, +0.529] |
| 2020* | 0.906 | 0.148 | 0.726 | 0.577 | 0.577 | +0.758 [-0.068, +1.314] |
| 2021* | 1.840 | 1.834 | 1.860 | 1.766 | 1.632 | +0.007 [-0.747, +0.720] |
| 2022* | -0.498 | -0.576 | -0.506 | -0.266 | -0.132 | +0.078 [-0.330, +0.572] |
| 2023* | 2.172 | 2.032 | 1.969 | 1.767 | 1.503 | +0.140 [-0.399, +0.632] |
| 2024* | 1.171 | 0.979 | 1.040 | 0.878 | 1.204 | +0.193 [-0.408, +0.804] |
| 2025* | 0.090 | 0.002 | 0.075 | 0.048 | -0.108 | +0.088 [-0.353, +0.546] |
| 2026* | 0.102 | -0.423 | 0.079 | 0.189 | 0.057 | +0.526 [+0.009, +0.991] |

## Per year: PPO ensemble vs VIX threshold rule (exploratory)

| Year | PPO ens. | VIX rule | PPO - VIX rule [95% CI] |
|---|---|---|---|
| 2019* | 0.229 | 0.324 | -0.095 [-0.560, +0.346] |
| 2020* | 0.906 | 0.755 | +0.151 [-0.042, +0.295] |
| 2021* | 1.840 | 1.860 | -0.020 [-0.476, +0.448] |
| 2022* | -0.498 | -0.181 | -0.317 [-0.971, +0.320] |
| 2023* | 2.172 | 2.058 | +0.114 [-0.408, +0.605] |
| 2024* | 1.171 | 1.040 | +0.132 [-0.290, +0.599] |
| 2025* | 0.090 | 0.487 | -0.397 [-0.904, +0.213] |
| 2026* | 0.102 | 0.602 | -0.500 [-1.257, +0.121] |

## Drawdown episodes (JUNIORBEES's deepest episode with its peak in the year)

| Year | Peak | Trough | Strategy | Peak-to-trough | Max DD in window |
|---|---|---|---|---|---|
| 2020 | 2020-01-24 | 2020-03-24 | PPO ensemble | -26.76% | -27.48% |
| 2020 | 2020-01-24 | 2020-03-24 | Vol targeting (floor 0.5) | -33.00% | -33.76% |
| 2020 | 2020-01-24 | 2020-03-24 | Vol targeting (no floor) | -33.00% | -33.76% |
| 2020 | 2020-01-24 | 2020-03-24 | VIX threshold rule | -29.25% | -30.05% |
| 2020 | 2020-01-24 | 2020-03-24 | Best fixed action | -16.79% | -17.19% |
| 2020 | 2020-01-24 | 2020-03-24 | Equal weight (monthly) | -36.52% | -36.98% |
| 2020 | 2020-01-24 | 2020-03-24 | JUNIORBEES buy-and-hold | -34.51% | -34.51% |

Delisting haircut sensitivity: not applicable (0 delisting exits on the path).

Figures: figures/equity, drawdown, exposure_vs_vix, per_year_delta_sharpe, per_year_delta_sharpe_vix_rule, seed_spread (png and pdf).
