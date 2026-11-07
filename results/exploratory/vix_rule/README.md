# VIX threshold rule (post-hoc, exploratory)

Not pre-registered. Designed after the test run showed PPO's exposure rising with India VIX, and fixed in `eval/vix_rule.py` (commit 3453d9f) before it was run: exposure 1.0 when India VIX at the decision close is at or above a threshold, else 0.5; threshold {15, 20, 25} x horizon {5, 21} chosen per fold on the two validation years. Logged in ledger.csv before evaluation and counted as a trial.

## Selected per fold

| fold | threshold | horizon | val_sharpe |
|---|---|---|---|
| 2019 | 20 | 21 | 0.883 |
| 2020 | 20 | 21 | 0.049 |
| 2021 | 25 | 21 | 0.929 |
| 2022 | 25 | 21 | 1.681 |
| 2023 | 20 | 21 | 1.184 |
| 2024 | 20 | 21 | 1.490 |
| 2025 | 15 | 5 | 1.579 |
| 2026 | 15 | 5 | 0.857 |

## Combined path 2019-01-01..2026-11-06

| strategy | sharpe | cagr | vol | max_drawdown | avg_exposure |
|---|---|---|---|---|---|
| PPO ensemble | 0.808 | 14.91% | 11.42% | -20.67% | 0.66 |
| VIX threshold rule (post-hoc) | 0.658 | 13.39% | 12.15% | -30.58% | 0.61 |
| Vol targeting (floor 0.5) | 0.488 | 11.37% | 12.89% | -29.89% | 0.87 |
| Risk parity, e=0.70 | 0.619 | 12.18% | 10.96% | -24.33% | 0.70 |

## Paired block bootstrap (same settings as H1)

| period | comparison | sharpe_a | sharpe_b | delta | p05 | ci | share_le_0 |
|---|---|---|---|---|---|---|---|
| 2019-2026 (all) | PPO ensemble - VIX threshold rule (post-hoc) | 0.808 | 0.658 | +0.151 | -0.176 | [-0.220, +0.417] | 0.258 |
| 2019-2026 (all) | VIX threshold rule (post-hoc) - Vol targeting (floor 0.5) | 0.658 | 0.488 | +0.169 | -0.070 | [-0.122, +0.535] | 0.119 |
| 2019-2021 (not previously seen) | PPO ensemble - VIX threshold rule (post-hoc) | 1.162 | 0.748 | +0.414 | -0.123 | [-0.182, +0.774] | 0.139 |
| 2019-2021 (not previously seen) | VIX threshold rule (post-hoc) - Vol targeting (floor 0.5) | 0.748 | 0.555 | +0.193 | -0.170 | [-0.250, +0.746] | 0.174 |
| 2022-2026 (previously seen) | PPO ensemble - VIX threshold rule (post-hoc) | 0.487 | 0.646 | -0.159 | -0.436 | [-0.489, +0.176] | 0.818 |
| 2022-2026 (previously seen) | VIX threshold rule (post-hoc) - Vol targeting (floor 0.5) | 0.646 | 0.443 | +0.203 | -0.069 | [-0.120, +0.531] | 0.113 |
| All years except 2020 | PPO ensemble - VIX threshold rule (post-hoc) | 0.649 | 0.742 | -0.092 | -0.317 | [-0.360, +0.183] | 0.739 |
| All years except 2020 | VIX threshold rule (post-hoc) - Vol targeting (floor 0.5) | 0.742 | 0.548 | +0.193 | -0.018 | [-0.060, +0.451] | 0.068 |

## Sharpe by year

| year | PPO ensemble | VIX threshold rule (post-hoc) | Vol targeting (floor 0.5) | Risk parity, e=0.70 |
|---|---|---|---|---|
| 2019 | 0.077 | 0.277 | 0.125 | 0.105 |
| 2020 | 1.491 | 0.841 | 0.317 | 0.774 |
| 2021 | 1.630 | 1.787 | 1.666 | 1.793 |
| 2022 | 0.308 | 0.505 | 0.025 | 0.144 |
| 2023 | 2.400 | 2.396 | 2.380 | 2.390 |
| 2024 | 0.539 | 0.273 | 0.259 | 0.279 |
| 2025 | 0.483 | 0.825 | 0.502 | 0.553 |
| 2026 | -0.379 | 0.113 | -0.573 | -0.522 |

## Agreement with PPO

Session-level exposure correlation with the PPO ensemble: +0.37. Share of sessions on the same side of 0.75 exposure: 0.74.

## Deflated Sharpe of the PPO ensemble with the ledger as it now stands

This study's trials: 2 (DSR 0.957); with recorded prior work: 84 (DSR 0.410). The protocol report computed these at the test run with one trial; quote the counts as of submission.
