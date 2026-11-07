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

## Data notes

- **Roster.** 87 symbols were in the index between 2014-01-01 and the freeze.
  - It is rebuilt from 76 archived copies of NSE's own constituent files, plus 28 dated changes, each with its notice (`data/roster/changes.csv`).
  - It matches 67 of 68 snapshots exactly. The one mismatch is a stale 2015 copy of the pre-rename file (`data/roster/validation.csv`).
- **Prices.** 81 symbols have usable Yahoo series. Six have none and are excluded during their membership: CAIRN, HDFC, IDFC, JPASSOCIAT, RANBAXY and TATAMTRDVR.
- **Corporate-action repairs** are logged with sources in `data/processed/price_repairs.csv`: three demergers (RIL 2023, ITC 2025, Tata Motors 2025) and one Yahoo split-adjustment artifact (Trent 2026). Other large moves are reviewed genuine events (`price_jumps_explained.csv`).
- **Benchmark.** NIFTYBEES adjusted close, with the Dec-2019 vendor glitch repaired (`benchmark_repairs.csv`).
