"""Append-only trial ledger (ledger.csv at the repo root).

Every configuration that is trained or selected gets one row, written automatically
by train/walk_forward.py, baselines/select.py and train/run_test.py. Seeds of one
configuration are one trial. The Deflated Sharpe Ratio takes its trial count from here.
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone

import pandas as pd

from config import ROOT, protocol_hash

LEDGER = ROOT / "ledger.csv"
COLUMNS = ["id", "time_utc", "git_sha", "protocol_sha256", "kind", "description",
           "folds", "seeds", "steps", "val_sharpe_mean", "touched_test", "counts_as_trial", "notes"]
KINDS = {"prior_work", "dev_smoke", "ppo_config", "baseline_grid", "test_run", "sensitivity"}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def read() -> pd.DataFrame:
    return pd.read_csv(LEDGER, dtype=str) if LEDGER.exists() else pd.DataFrame(columns=COLUMNS)


def append(kind: str, description: str, *, folds: str = "", seeds: str = "", steps: str = "",
           val_sharpe_mean: float | str = "", touched_test: bool = False,
           counts_as_trial: bool = True, notes: str = "") -> int:
    if kind not in KINDS:
        raise ValueError(f"unknown ledger kind {kind!r}")
    df = read()
    new_id = int(df["id"].astype(int).max()) + 1 if len(df) else 1
    row = {"id": new_id, "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "git_sha": _git_sha(), "protocol_sha256": protocol_hash()[:16], "kind": kind,
           "description": description, "folds": folds, "seeds": seeds, "steps": steps,
           "val_sharpe_mean": val_sharpe_mean if val_sharpe_mean == "" else f"{float(val_sharpe_mean):.4f}",
           "touched_test": str(bool(touched_test)), "counts_as_trial": str(bool(counts_as_trial)),
           "notes": notes}
    new_file = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return new_id


def trial_counts() -> dict[str, int]:
    """Trials for the DSR: this study only, and this study plus recorded prior work."""
    df = read()
    if df.empty:
        return {"this_study": 0, "with_prior": 0}
    trial = df[df["counts_as_trial"] == "True"]
    study = trial[trial["kind"].isin(["ppo_config", "dev_smoke"])]
    prior = trial[trial["kind"] == "prior_work"]
    n_prior = int(pd.to_numeric(prior["notes"].str.extract(r"n_trials=(\d+)")[0]).fillna(0).sum())
    return {"this_study": len(study), "with_prior": len(study) + n_prior}
