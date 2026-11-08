# RL as a risk-budget controller (Nifty 50, walk-forward 2019-2026)

PPO chooses only the portfolio's exposure (50-100%) and rebalance horizon (5 or 21 sessions). An equal-risk-contribution allocator picks the stocks from a point-in-time Nifty 50 universe. Cash earns the RBI 91-day T-bill rate. The primary test is against volatility targeting on the same engine. See `protocol.md` for the pre-registered design. The v1 PPO-MVO code is in `legacy/`.

## Layout

| Path | What it does |
|---|---|
| `protocol.md`, `config/protocol.yaml` | The pre-registered protocol. Code reads only the yaml |
| `ledger.csv` | Append-only trial ledger, used for the Deflated Sharpe trial count |
| `data/` | Builds the roster, prices, market series, risk-free rate and panel. `MANIFEST.json` holds the hashes |
| `env/` | Risk-parity allocator, causal features, the portfolio engine, the Gym wrapper |
| `baselines/` | Buy-and-hold, equal-weight, fixed exposure, vol targeting, random, best fixed action |
| `train/` | Walk-forward runner, ensemble, gated test run |
| `eval/` | Metrics, paired block bootstrap, PSR/DSR, report |
| `tests/` | Leakage, timing, cost, risk-parity and statistics tests |

## Rebuild from scratch

```bash
python -m venv ~/.venvs/rl-risk-budget && ~/.venvs/rl-risk-budget/bin/pip install -r requirements.txt
python -m data.roster fetch        # archived NSE constituent files -> snapshots
python -m data.roster_build        # dated changes -> daily membership, validated
python -m data.market              # index, NIFTYBEES, India VIX, USD/INR
python -m data.prices              # constituent prices, corporate-action repairs, QA
python -m data.rf                  # RBI 91-day T-bill (file in data/raw/rbi/)
python -m data.panel               # engine arrays (risk-parity weights, features)
python -m data.manifest --verify   # hashes
pytest -q
```

## Status (2026-11-07): protocol complete

1. Risk-free series built from RBI's 91-day T-bill auction table (3762ab8).
2. Protocol signed by all three authors and tagged `protocol-v1` (9b4c81e).
3. Protocol training, 8 folds x 5 seeds x 200k steps, committed in `runs/protocol/` (30e9463).
4. The single test run (20a1adb): `results/report.md`, with tables in `results/tables/` and figures in `results/figures/`.

Post-unblinding analyses are in `results/exploratory/` (`python -m eval.exploratory`) and are labelled exploratory. Any further analysis must be logged in `ledger.csv`. Use kind `exploratory` for re-reading the test returns, or `sensitivity` for any new strategy variant, which counts as a trial in the Deflated Sharpe.

## Protocol v2 (draft): the frozen pipeline on the Nifty Next 50

`protocol_v2.md` and `config/protocol_v2.yaml` are a draft for the authors to sign. They rerun v1 unchanged on the point-in-time Nifty Next 50 and register v1's post-hoc VIX threshold rule as a comparator. The two primary hypotheses, H1 (beats vol targeting) and H2 (beats the VIX rule), are tested under Holm.

Every command reads the active protocol from `RB_PROTOCOL`. Without it, v1 is used and nothing v1 produced changes. v2 keeps its data, runs and results under `studies/next50/`.

```bash
export RB_PROTOCOL=protocol_v2.yaml
python -m data.roster_next50 notices      # NSE Indices notices: listing, PDFs (cached in scratch/), text
python -m data.roster_next50 sections     # Next 50 rows parsed from each notice, plus the hand-review queue
python -m data.roster_next50 snapshots    # archived Next 50 and Nifty 100 lists
python -m data.roster_next50 build        # changes.csv -> daily membership, validated against every archived list
python -m data.market && python -m data.rf && python -m data.prices && python -m data.panel
python -m data.manifest --verify          # studies/next50/data/MANIFEST.json
```

Status: registered as `protocol-v2` on 2026-11-08 (all three authors signed and kept the defaults in Section 9). Trained in bdac9c2 (ledger row 12) and tested once in 40b0c4b (ledger row 13).

**Result: neither hypothesis supported**, so under the signed decision table the v1 result does not replicate.
- H1 (PPO ensemble minus floored vol targeting): +0.195, 95% CI [-0.088, +0.447], p = 0.085 against a Holm threshold of 0.025.
- H2 (PPO ensemble minus the VIX threshold rule): -0.028, p = 0.61.

The report is in `studies/next50/results/report.md`. The manuscript reports both studies.

Post-test data audit (ledger row 15, `studies/next50/results/repair_audit.csv`): the automatic split-boundary repair inherited from v1 fired wrongly once. On 2024-06-04, a genuine -19.2% election-day fall in CONCOR matched a later split factor and was repaired to +1.0%. CONCOR had left the index in March 2021 and was never eligible within reach of that day, so no result is affected, and the registered data are unchanged. Any future protocol should allow that repair only on the first session of a calendar year, which is where the vendor artifact occurs.

## Paper

`paper/main.tex` is the working draft. Every number and table in it is generated from the committed results, so rebuild both together:

```bash
python -m eval.paper_numbers                 # writes paper/generated/ (macros and tables)
cd paper && tectonic -X compile main.tex     # writes paper/main.pdf
```

Figures come straight from `results/figures/`. Check `paper/refs.bib` against the publishers before submission.

## Data notes

- **Roster.** 87 symbols were in the index between 2014-01-01 and the freeze.
  - It is rebuilt from 76 archived copies of NSE's own constituent files, plus 78 add/remove events on 25 effective dates, each with its source (`data/roster/changes.csv`).
  - It matches 67 of 68 snapshots exactly. The one mismatch is a stale 2015 copy of the pre-rename file (`data/roster/validation.csv`).
- **Prices.** 81 symbols have usable Yahoo series. Six have none and are excluded during their membership: CAIRN, HDFC, IDFC, JPASSOCIAT, RANBAXY and TATAMTRDVR.
- **Corporate-action repairs** are logged with sources in `data/processed/price_repairs.csv`: three demergers (RIL 2023, ITC 2025, Tata Motors 2025) and one Yahoo split-adjustment artifact (Trent 2026). Other large moves are reviewed genuine events (`price_jumps_explained.csv`).
- **Benchmark.** NIFTYBEES adjusted close, with the Dec-2019 vendor glitch repaired (`benchmark_repairs.csv`).
