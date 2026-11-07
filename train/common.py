"""Shared pieces: fold windows, policy wrappers, the protocol gate."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from config import ROOT, load, protocol_hash
from env.engine import MarketData
from env.features import zscore_stats
from env.gym_env import build_obs

RUNS = ROOT / "runs"


@dataclass
class Fold:
    test_year: int
    train_lo: int      # decision close that starts training (last session of 2013)
    train_hi: int      # last training session (end of Y-3)
    val: list          # [(year, start_idx, end_idx)] for Y-2 and Y-1
    test: tuple        # (start_idx, end_idx)
    mu: np.ndarray
    sd: np.ndarray


def folds(data: MarketData, years: list[int] | None = None) -> list[Fold]:
    cfg = load()["folds"]
    out = []
    for Y in (years or cfg["test_years"]):
        lo = data.last_session_before(cfg["train_start_year"])
        hi = data.last_session_of(Y - 3)
        vals = [(y, data.last_session_before(y), data.last_session_of(y))
                for y in range(Y - cfg["n_val_years"], Y)]
        test = (data.last_session_before(Y), data.last_session_of(Y))
        import pandas as pd
        mu, sd = zscore_stats(pd.DataFrame(data.feats), lo, hi)
        out.append(Fold(Y, lo, hi, vals, test, mu, sd))
    return out


class PPOPolicy:
    """Deterministic SB3 policy as an engine policy (uses its own fold's z-score stats)."""

    def __init__(self, model, data: MarketData, fold: Fold):
        cfg = load()
        self.model, self.data, self.fold = model, data, fold
        self.E = cfg["portfolio"]["exposure_grid"]
        self.H = cfg["portfolio"]["horizon_grid"]
        self.clip = cfg["state"]["clip"]

    def action(self, sim):
        obs = build_obs(self.data, sim, self.fold.mu, self.fold.sd, self.clip)
        a, _ = self.model.predict(obs, deterministic=True)
        return self.E[int(a[0])], self.H[int(a[1])]

    def __call__(self, sim, t):
        return self.action(sim)


class Ensemble:
    """Mean exposure and median horizon of several policies, queried on the ensemble's own state."""

    def __init__(self, members: list):
        self.members = members

    def __call__(self, sim, t):
        es, hs = zip(*(m(sim, t) for m in self.members))
        return float(np.mean(es)), int(np.median(hs))


class FoldSwitch:
    """Use the policy of the fold whose test year the decision governs (year of session t+1)."""

    def __init__(self, data: MarketData, by_year: dict):
        self.data, self.by_year = data, by_year

    def __call__(self, sim, t):
        y = int(self.data.dates[min(t + 1, len(self.data.dates) - 1)].year)
        return self.by_year[y](sim, t)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def protocol_gate() -> list[str]:
    """Reasons the protocol-mode run must not start (empty list = allowed)."""
    from data.rf import is_provisional
    problems = []
    if not git("tag", "-l", "protocol-v1"):
        problems.append("tag protocol-v1 does not exist (protocol not signed off)")
    else:
        diff = subprocess.run(["git", "diff", "--quiet", "protocol-v1", "--", "config/protocol.yaml"], cwd=ROOT)
        if diff.returncode != 0:
            problems.append("config/protocol.yaml changed since protocol-v1")
    if git("status", "--porcelain", "--untracked-files=no"):
        problems.append("working tree has uncommitted changes")
    if is_provisional():
        problems.append("risk-free series is provisional (add the RBI file and rebuild)")
    return problems


def run_tag() -> str:
    return f"{git('rev-parse', '--short', 'HEAD')}-{protocol_hash()[:8]}"
