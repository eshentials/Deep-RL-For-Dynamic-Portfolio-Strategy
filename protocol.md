# Pre-registered protocol: RL as a risk-budget controller

**Status: DRAFT.** This becomes binding when every author has signed below and the commit is tagged `protocol-v1`. Until then, no strategy may be scored on any test year (2019 onward). `train/run_test.py` enforces this: it refuses to run unless the tag exists, `config/protocol.yaml` is unchanged since the tag, and the working tree is clean.

The machine-readable version is `config/protocol.yaml`. The code reads only that file. At tagging, its SHA-256 is recorded here: `(filled at tag time)`. If this document and the yaml ever disagree, the yaml at the tag governs, and the disagreement is reported as a deviation.

## 1. Question and hypotheses

The previous version of this project used PPO to pick the risk aversion and the holding horizon of a mean-variance portfolio. It did not beat the passive benchmark: a return-forecasting controller failed. This study moves the agent to risk control only. A fixed allocator (equal risk contribution) decides *what* to hold, and the agent decides *how much* (exposure) and *for how long* (horizon).

- **H1 (primary).** Over the combined daily path of all test years (2019-01-01 to the data freeze), the PPO ensemble has a higher annualised Sharpe ratio than the volatility-targeting rule (floored at 0.5 exposure, Section 6).
  - Test: paired circular block bootstrap of the Sharpe difference (20-session blocks, 10,000 draws, seed 20261107).
  - H1 is supported only if the 5th percentile of the bootstrap distribution of Sharpe(PPO) minus Sharpe(vol targeting) is above zero (one-sided, α = 0.05).
- **Secondary outcomes.** Maximum drawdown and Calmar ratio of the PPO ensemble against equal-weight Nifty 50 and NIFTYBEES, with paired bootstrap intervals.
- Every other comparison is **exploratory** and labelled as such.

## 2. Data

- **Universe.** A point-in-time Nifty 50 roster, rebuilt from dated archive copies of NSE's own constituent files: `ind_niftylist.csv` (2013-2015) and `ind_nifty50list.csv` (2016 onward), plus NSE's index API.
  - Each change is dated from NSE or niftyindices notices and carries a confidence flag (`data/roster/changes.csv`).
  - The rebuilt roster must match every archived copy on its capture date, and must hold exactly 50 members on every session apart from documented demerger placeholder days.
- **Eligibility on session t.** The stock is a member on t, and has a complete 120-session window of returns ending at t.
- **Prices.** Yahoo Finance adjusted closes. A symbol is used only if, during its membership:
  - it has a price on at least 98% of sessions, and
  - it has no unexplained day with |return| > 25%.

  Symbols failing either test are excluded for their membership months and listed in the data appendix. This is a stated limitation.
- **Delistings.** The position exits at the last traded price. A sensitivity check applies a -30% loss at those exits.
- **Benchmark.** NIFTYBEES adjusted close (total return).
  - A vendor glitch, meaning an ETF/index level ratio more than 20% off its trailing median, is replaced by the previous level times index growth, and logged.
  - Genuine tracking gaps, such as ETF premiums in crashes, are kept and logged.
- **Risk-free rate.** RBI 91-day Treasury bill primary auction cut-off yield (weekly).
  - Each yield is carried forward from its result date to later sessions.
  - Cash accrues (1 + y)^(calendar days / 365) - 1 between sessions.
  - Cash in every strategy earns this rate, and every Sharpe ratio uses excess returns over it.
- **Data freeze.** The last NSE session on or before 2026-11-06.
- **Hashes.** Every raw and processed file is hashed in `data/MANIFEST.json`. `python -m data.manifest --verify` must pass at the tag.

## 3. Portfolio and execution

- **Allocator.** Equal risk contribution (Spinu convex formulation) over eligible members, using a Ledoit-Wolf covariance of the trailing 120 sessions. Weights are computed once per session, independent of the agent.
- **Portfolio.** Exposure e times the risk-parity weights. The remaining (1 - e) is held in cash at the risk-free rate.
- **Execution.**
  - Decide and trade at the close of session t. The new portfolio earns returns from t+1.
  - Holdings drift between decisions.
  - A stock leaving the index is sold at the close of its last member session, and the proceeds sit in cash until the next decision.
- **Costs.** 10 bp times one-way turnover (the sum of |Δw|), charged against NAV, including forced sales.
- **Year boundary.** A decision is forced at the first session of each calendar year. Every test year is then governed only by its own fold's policy, and the same applies to every baseline.

## 4. The agent

- **Action.** MultiDiscrete([4, 2]):
  - exposure e ∈ {0.5, 0.7, 0.85, 1.0}
  - horizon h ∈ {5, 21} sessions
- **State.** 13 features, none of them per stock, all computed from data up to close t:

  | Group | Features |
  |---|---|
  | Index volatility | realised volatility over 20 and 60 sessions, and their ratio |
  | Index momentum | log return over 21 and 63 sessions |
  | Index drawdown | distance from the index's 252-session high |
  | Correlation | average pairwise correlation of eligible members over 60 sessions |
  | India VIX | level, and its 5-session log change |
  | Currency | USD/INR 21-session log change, lagged one session (Yahoo's FX close falls after the NSE close) |
  | The agent's own portfolio | current exposure, own drawdown, own 20-session volatility |

  Market features are z-scored with each fold's training-window statistics only, then clipped to ±5.
- **Reward per holding interval.** The sum of daily excess log returns (net of costs) minus κ times max(0, drawdown at interval end minus drawdown at interval start). κ = 1 is fixed and not tuned. Rewards are scaled ×100.
- **Training.**
  - Stable-Baselines3 PPO: MLP [64, 64], learning rate 3e-4, n_steps 1024, batch 64, 10 epochs, γ = 1, GAE λ = 0.95, clip 0.2, entropy 0.01.
  - 5 seeds (0-4) × 200,000 steps per fold.
  - Episodes are random 252-session windows inside the training years, each starting in cash.
- **Checkpoint selection.** Every 10,000 steps, the policy runs deterministically on each validation year, starting in cash. The checkpoint with the highest mean of the two validation Sharpe ratios is kept. Test data never enters selection.
- **Ensemble (primary strategy).** At each of its own decision dates, the ensemble queries all 5 seeds on its own state. Exposure is the mean of their exposures, and horizon is the median of their horizons. Each seed is also reported separately.

## 5. Walk-forward folds

| Test year | Train | Validate |
|---|---|---|
| 2019 | 2014-2016 | 2017-2018 |
| 2020 | 2014-2017 | 2018-2019 |
| 2021 | 2014-2018 | 2019-2020 |
| 2022 | 2014-2019 | 2020-2021 |
| 2023 | 2014-2020 | 2021-2022 |
| 2024 | 2014-2021 | 2022-2023 |
| 2025 | 2014-2022 | 2023-2024 |
| 2026 | 2014-2023 | 2024-2025 (test runs to the freeze date) |

**Previously seen.** Earlier analyses in this project (7 fixed stocks) examined 2022-2026. Those test years are labelled "previously seen" in every table. The headline claim rests on the combined 2019-2026 path.

## 6. Baselines

All baselines use the same engine, costs, start date (in cash, 2019-01-01) and year-boundary rule as the agent.

1. **NIFTYBEES** buy-and-hold.
2. **Equal-weight** eligible members, rebalanced on the first session of each month.
3. **Risk parity at fixed exposure**, e ∈ {0.5, 0.7, 0.85, 1.0}, rebalanced every 21 sessions.
4. **Volatility targeting, the primary comparator.**
   - Exposure e = clip(σ* / σ̂60, 0.5, 1.0), where σ̂60 is the 60-session realised volatility of the unlevered risk-parity portfolio.
   - The floor of 0.5 gives the rule the same exposure range as the agent, so the comparison isolates learned control from rule-based control.
   - The target σ* ∈ {10, 12, 15}% and the horizon ∈ {5, 21} are chosen per fold on that fold's validation years.
   - The unfloored rule, clip(σ*/σ̂60, 0, 1), is reported as a secondary baseline.
5. **Random policy** over the same 8 actions. 100 seeds, reported as the median with the 5-95% band.
6. **Best fixed action** (e, h), chosen per fold on validation.

A2C with an identical setup is cut from this protocol. The code keeps the hook, and adding it later requires a new protocol version.

## 7. Statistics and reporting

- **Metrics.** Annualised Sharpe ratio of daily excess returns (×√252), CAGR, volatility, maximum drawdown, Calmar (CAGR / |MDD|), average exposure, turnover and total costs.
- **H1.** As in Section 1.
  - Paired 95% intervals are reported for every other Sharpe difference, all labelled exploratory.
  - Max drawdown and Calmar intervals use the same bootstrap.
- **Deflated Sharpe Ratio** (Bailey and López de Prado, 2014) of the PPO ensemble on the combined test path.
  - SR₀ uses the null variance of the Sharpe estimator, 1/(T-1).
  - It is reported for two trial counts N: this study's ledger trials, and this study plus the recorded prior work (`ledger.csv`). The Probabilistic Sharpe Ratio (N = 1) is also reported.
- **Per-fold table.** One row per test year for the ensemble, each seed and each baseline, with previously seen years flagged.
- **Drawdown episodes.** For 2020 and 2025, take the deepest NIFTYBEES peak-to-trough episode whose peak falls in that year, found mechanically. Every strategy's drawdown over those exact dates is compared.
- **Pre-registered sensitivities**, all secondary:
  - execution one session later (close t+1)
  - 20 bp costs
  - the unfloored vol-targeting rule
  - the -30% delisting loss, where any eligible symbol stopped trading while a member

## 8. Correctness checks required before any training on the protocol

- **Future invariance.** Perturbing all data after a date t* leaves every feature, risk-parity weight, normalisation statistic, decision and NAV up to t* bit-identical.
- **Positive control.** An oracle that sees the next interval's return must show an implausibly high Sharpe, and the same rule lagged by one interval must not.
- **Hand checks.** Hand-computed cost trades, risk-parity identities (a diagonal covariance gives inverse-volatility weights), and the bootstrap and DSR unit tests.
- **Development data.** All development runs use only 2014-2018 (fold 2019's training and validation years).

## 9. Decision rules

- **If H1 is not supported,** the paper reports a careful comparison showing that RL does not beat simple volatility targeting, with the same tables. No new variant is run to change that outcome.
- **Post-unblinding runs.** Any analysis after the test run is labelled exploratory and logged in `ledger.csv`.
- **Trial ledger.** Every configuration trained or selected is appended to `ledger.csv` automatically. Seeds of one configuration are one trial. Each test run is also logged, so a test can't be rerun silently.
- **Deviations.** A change after tagging requires a new tag (`protocol-v2`) with a written reason. Results under v1 are still reported.

## 10. Known limitations at registration

- Symbols without usable adjusted prices are excluded for their membership months.
- Roster dates that rest on news reports rather than NSE notices are flagged as low confidence.
- The study covers one market and one price vendor.
- κ is fixed rather than tuned.
- The A2C ablation and the Nifty Next 50 fresh-market test are deferred.
- Fold 2019 trains on only three years.

## Sign-off

Each author adds their name and date. The tag is created only after all authors have signed.

| Author | Date |
|---|---|
| | |
