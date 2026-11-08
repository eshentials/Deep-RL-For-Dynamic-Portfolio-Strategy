# Pre-registered protocol v2: the frozen pipeline on the Nifty Next 50

**Status: DRAFT, not registered.** It becomes protocol v2 when all three authors sign below. Then `config/protocol_v2.yaml` is committed unchanged, its SHA-256 is recorded here, and the tag `protocol-v2` is created. No strategy has been scored on any Nifty Next 50 session from 2019 onward, and none will be before the tag. The gated runs (`train/walk_forward.py --protocol` and `train/run_test.py --protocol`, run with `RB_PROTOCOL=protocol_v2.yaml`) refuse to start without the tag, with a changed yaml, or with a dirty tree.

The machine-readable version is `config/protocol_v2.yaml`, and the code reads only that file. If this document and the yaml disagree, the yaml at the tag governs.

## 0. Why a second protocol

Protocol v1 tested PPO as a risk-budget controller on the point-in-time Nifty 50, 2019 to 2026. **H1 was supported:** the PPO ensemble's Sharpe ratio beat floored volatility targeting by +0.320, with a bootstrap 5th percentile of +0.054.

Three exploratory analyses run after unblinding (ledger rows 8 to 10) weaken that result:

- The margin rests on 2020. Excluding 2020, the difference is +0.101 and its interval includes zero.
- The ensemble's exposure tracks India VIX (correlation +0.62). It raises exposure after fear spikes.
- A simple rule written after seeing that behaviour is statistically indistinguishable from PPO. The rule holds full exposure when VIX is at or above a threshold and half exposure otherwise. The difference is +0.151, 95% CI [-0.220, +0.417], and in 2020 the two differ on a single decision.

None of these findings was pre-registered, so none can be claimed as confirmatory. Protocol v2 asks the two open questions out of sample, on stocks none of the design touched, with every choice fixed in advance:

1. Does the v1 result replicate on a different set of stocks?
2. Does PPO beat the simplest rule that describes what it appears to do?

**What a v2 result can and cannot show.** The stocks are new, but the years are not. The authors examined the 2019 to 2026 market path in v1, including the 2020 crash and the VIX series, and the two indices move together (daily return correlation 0.83 over 2014 to 2018). A positive v2 result shows that the v1 result is not specific to the Nifty 50's constituents. It says nothing about years after 2026. The VIX rule was designed after looking at v1's PPO, and that is why it is the right comparator: registering it unchanged, before any Next 50 test data is scored, is what makes H2 a fair test.

## 1. Hypotheses

- **H1 (replication).** On the combined daily path from 2019-01-01 to the freeze, the PPO ensemble has a higher annualised Sharpe ratio than volatility targeting floored at 0.5.
- **H2 (more than a VIX timer).** On the same path, the PPO ensemble has a higher annualised Sharpe ratio than the VIX threshold rule (Section 4).
- **Test.** For each hypothesis, a one-sided paired circular block bootstrap of the Sharpe difference: 20-session blocks, 10,000 draws, seed 20261107, as in v1. Its p-value is the share of draws at or below zero.
- **Multiplicity (Holm, family-wise α = 0.05).** The smaller p must be at most 0.025. If it is, the larger p must be at most 0.05. A hypothesis counts as supported only if this procedure rejects it.
- **Secondary outcomes.** Maximum drawdown and Calmar ratio of the PPO ensemble against equal weight and JUNIORBEES, with paired bootstrap intervals.
- Every other comparison is exploratory and labelled as such. Failing to reject a hypothesis is not evidence that the two strategies are equivalent.

**Decision table, fixed now:**

| H1 | H2 | What the paper reports |
|---|---|---|
| Supported | Supported | The edge over volatility targeting replicates on new stocks, and PPO beats the simplest rule describing its behaviour. |
| Supported | Not supported | The edge replicates, but there is no evidence that PPO does more than a VIX threshold rule. The paper describes the agent as a learned VIX timer. |
| Not supported | Supported | Reported as is. v1's H1 stands as a result on one universe. |
| Not supported | Not supported | The v1 result does not replicate on the Nifty Next 50. The abstract says so, and v1's H1 is reported as holding on one universe only. |

## 2. What is frozen and what changes

Everything below is identical to protocol v1. `tests/test_protocol_v2.py` fails if the v2 yaml changes any other key:

- the allocator, costs and execution;
- the action grid, reward (κ = 1), state features and z-scoring;
- the PPO settings, seeds, steps, checkpoint selection and ensemble rule;
- the folds, test years and freeze date;
- every v1 baseline and its selection grid, including the vol-targeting floor;
- the bootstrap, the DSR and the pre-registered sensitivities;
- the risk-free series.

Only these differ, and each is marked `v2:` in the yaml:

| Item | Protocol v1 | Protocol v2 |
|---|---|---|
| Universe | Point-in-time Nifty 50 | Point-in-time Nifty Next 50 (ranks 51 to 100 of the Nifty 100) |
| Index features | `^NSEI` | `^NSMIDCP` (Nifty Next 50), on the same `^NSEI` session calendar |
| Average pairwise correlation | Eligible Nifty 50 members | Eligible Next 50 members |
| Benchmark | NIFTYBEES | JUNIORBEES (the Nifty Next 50 ETF) |
| VIX threshold rule | Post-hoc, exploratory (ledger row 9) | Registered comparator, grid unchanged |
| Primary hypotheses | H1 | H1 and H2 under Holm |
| Years labelled "previously seen" | 2022 to 2026 | All test years (v1 examined 2019 to 2026) |

## 3. Data, as built for registration

**Roster.** Rebuilt from NSE Indices notices (`data/roster_next50.py`).

- All 449 equity notices on niftyindices.com from October 2013 to the freeze were downloaded and their text searched for the index. Two failed, both about the Nifty IPO index.
- The changes come from 38 notices:
  - all 26 semi-annual reviews, March 2014 to September 2026 (NSE deferred the March 2020 review because of COVID-19 and held it in June 2020);
  - 12 ad-hoc changes, for mergers, demergers and Nifty 50 replacements.
- That gives 308 dated changes on 37 dates (`studies/next50/data/roster/changes.csv`).
- Every hand decision is listed in `hand_decisions.csv`: the deferred review skipped, three effective dates resolved, and one image-only notice read from its rendered pages.

**Validation.**

- The roster matches all 37 archived readings from 2014-01-22 to 2026-09-30: 22 copies of NSE's Next 50 file, and 15 copies of the Nifty 100 file minus v1's validated Nifty 50.
- It holds exactly 50 members on every session and never overlaps v1's Nifty 50.
- Two reading rules are documented:
  - Demerger placeholders are dropped inside their notice-dated windows, as in v1: Siemens Energy in 2025 and four Vedanta entities in 2026.
  - Tata Motors' DVR share is dropped from Nifty 100 readings. It is a 101st line in 2017 to 2019 copies of that file but never appears in a Next 50 copy.
- Archive copies are fetched only when the archive's Memento-Datetime equals the requested capture, because the archive once served a 2026 copy for a 2024 capture.

**Prices.** The vendor and rules are the same as v1: Yahoo adjusted closes, at least 98% coverage over membership, and no unexplained day beyond ±25%.

- 152 symbols were members between 2014-01-01 and the freeze, and 143 are usable.
- Nine are excluded for their membership:
  - eight with no Yahoo series: ABIRLANUVO, CAIRN, GSKCONS, INGVYSYABK, MINDTREE, PEL, RANBAXY and RELCAPITAL;
  - ICICIPRULI, whose Yahoo history now starts in July 2026, although NSE's bhavcopy shows continuous trading under the same symbol and ISIN.
- Together they are 2.8% of member-sessions in the test years (3.9% since 2014). PEL (to August 2022) and ICICIPRULI (to September 2025) are the material gaps. In v1 the material gap was HDFC.

**Exchange check (new in v2).** A day beyond ±25% is accepted only if it meets both conditions:

1. NSE's official bhavcopy gives the same return as the vendor, within 1.5 percentage points.
2. A reviewed cause is recorded.

All 13 such days passed (`price_jump_checks.csv`). Demergers are repaired rather than confirmed, because NSE's previous close is not adjusted for a demerger. Vedanta's 2026-04-30 ex-date uses the special pre-open price of 289.50 from NSE's bhavcopy.

**Market series.**

- The sessions are v1's.
- `^NSMIDCP` lacks one session (2025-02-01), which is filled forward.
- JUNIORBEES has one missing day (2025-10-24), imputed from index growth. Six tracking gaps are kept and logged.
- India VIX and USD/INR equal v1's series to within 4e-15.
- The risk-free file is byte-identical to v1's.
- Between 45 and 50 stocks are eligible on every session.

**Hashes.** `studies/next50/data/MANIFEST.json` lists 516 files. `RB_PROTOCOL=protocol_v2.yaml python -m data.manifest --verify` must pass at the tag.

## 4. Baselines

These are v1's baselines, on the same engine, costs, start in cash and year-boundary rule, with JUNIORBEES in place of NIFTYBEES. One baseline is added:

**7. VIX threshold rule (registered comparator).**

- **Rule.** Exposure is 1.0 when India VIX at the decision close is at or above a threshold θ, and 0.5 otherwise.
- **Selection.** θ ∈ {15, 20, 25} and the horizon ∈ {5, 21} are chosen per fold on that fold's two validation years. This is the same selection used for every tuned baseline.
- **Provenance.** The grid was fixed in commit 3453d9f, before the rule was ever run. The rule now lives in `baselines/rules.py`, and a test shows that it is the same function v1's post-hoc analysis used.
- **Sensitivities.** Its one-day-lag and 20 bp versions are reported with the other sensitivities.

**A known property of the frozen comparator.** In the development years (2017 to 2018), the selected vol-targeting rule sat near full exposure, at an average of 0.98. Its 10 to 15% targets exceed the risk-parity book's own volatility most of the time. H1 therefore mainly tests PPO against fully invested risk parity that de-risks in volatile spells. H2 is the sharper test. This is unchanged from v1 by design.

## 5. Statistics and reporting

These are as in v1 Sections 5 and 7, with three additions:

- The per-year table adds PPO minus the VIX rule (exploratory).
- The Deflated Sharpe Ratio takes its trial count from the project ledger across both protocols, which is conservative.
- Every test year is flagged "previously seen".

## 6. Use of Next 50 data before registration

- The data were built for the full period (roster, prices and checks). These are data checks, not strategy scores.
- One development run was made (ledger row 11): fold 2019, 5 seeds × 200,000 steps, trained on 2014 to 2016 and validated on 2017 to 2018.
- The test and report code was dry-run on 2017 to 2018 only.
- Both ran after the v2 yaml was committed (50e47d9). Nothing from them changed this protocol, apart from the comparator note in Section 4.
- No strategy has been scored on any session from 2019 onward.

## 7. Decision rules

These are as in v1 Section 9. In addition, whatever v2 shows, it is reported in the paper. No further universe is tried to rescue a negative result, and any further universe needs its own protocol and sign-off.

## 8. Known limitations at registration

- The years are the same as v1's, so this is not a test of a new period.
- The two universes share the market factor (correlation 0.83), and the VIX path is identical.
- Nine symbols are excluded, covering 2.8% of member-sessions in the test years.
- There is one price vendor. NSE's bhavcopy is used only to check flagged days and to price one demerger.
- A Next 50 change announced only outside niftyindices.com's press releases would be missed. All 37 archived readings match, the latest on 2026-09-30.
- κ is fixed, and A2C is not run.

## 9. Choices for the authors before signing

The draft takes the first option in each case.

1. **Vol-targeting grid.** Keep v1's grid (frozen), or scale the targets to each universe's volatility. Scaling makes the comparator stronger but breaks the frozen pipeline.
2. **Price gaps.** Keep v1's rule, or fill the nine excluded symbols from NSE bhavcopy closes. Filling gives fewer exclusions but adds a second price source that v1 did not use.
3. **Multiplicity.** Holm across H1 and H2, or a fixed sequence: test H1 at 0.05, and test H2 at 0.05 only if H1 holds. The fixed sequence gives each hypothesis the full α but leaves H2 untested if H1 fails.

## Sign-off

Each author adds their name and date. The tag is created only after all authors have signed.

| Author | Date |
|---|---|
| Eshani Parulekar | |
| Dhruv Pande | |
| Rishaan Damani | |

At registration: `config/protocol_v2.yaml` SHA-256 `(recorded at tagging)`, tag `protocol-v2`.

After the tag:

```
RB_PROTOCOL=protocol_v2.yaml python -m train.walk_forward --protocol   # 8 folds x 5 seeds x 200k steps (about 25 min)
RB_PROTOCOL=protocol_v2.yaml python -m train.run_test --protocol       # the single test run (ledger row written first)
```
