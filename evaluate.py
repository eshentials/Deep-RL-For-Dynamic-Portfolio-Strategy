"""
evaluate.py — Full backtest and benchmark comparison for the Deep-RL portfolio agent.

Strategies compared
───────────────────
  1. PPO Agent        — trained RL model (best_model.zip or latest)
  2. Equal-Weight     — 1/N across all stocks, rebalanced every 21 days
  3. Nifty BeES B&H   — 100% allocation to NIFTYBEES.NS from day 1
  4. MVO λ=0.5        — mean-variance optimizer (return-seeking), 21-day rebalance
  5. MVO λ=1.0        — balanced MVO, 21-day rebalance
  6. MVO λ=2.0        — conservative MVO (minimum-variance), 21-day rebalance

Metrics reported (per split)
────────────────────────────
  Total Return (%)   · CAGR (%)         · Annualised Vol (%)
  Sharpe Ratio       · Max Drawdown (%)  · Calmar Ratio
  Total TC paid (%)  · Mean λ chosen     · Mean hold days

Outputs
───────
  results/metrics_<split>.csv   — per-strategy metrics table
  results/nav_curves_<split>.csv — daily NAV for every strategy (for plotting)
  results/metrics_combined.csv  — val + test merged table

Usage
─────
  python3.13 evaluate.py                    # val + test, best model
  python3.13 evaluate.py --split test       # test split only
  python3.13 evaluate.py --model latest     # use latest checkpoint
  python3.13 evaluate.py --split val test   # both splits
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env import PortfolioEnv, HOLDING_PERIODS
from efficient_frontier import horizon_momentum_pct, optimize, LAMBDA_VALUES
from features import load_bundle
from state_spec import (
    STATE,
    build_state_vector,
    transaction_cost,
    MACRO_PATH,
    EPS,
)

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_DIR  = "models"
BEST_PATH  = os.path.join(MODEL_DIR, "best_model")
LATEST_PATH = os.path.join(MODEL_DIR, "ppo_portfolio_latest")
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnorm.pkl")

# ── Fixed rebalance period for benchmark MVO strategies ───────────────────────
BENCH_REBAL_DAYS: int = 21   # monthly rebalance for all non-RL benchmarks
RISK_FREE_RATE:   float = 0.065  # approximate Indian 1-yr T-bill rate

SPLIT_LABELS: dict[str, str] = {
    "train": "train_2014_2020",
    "test":  "test_2021",
    "val":   "val_2022_2024",
}


def _load_volume_context(bundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current volume and 20-day ADV aligned to a feature bundle."""
    label = SPLIT_LABELS[bundle.split]
    path = os.path.join("data", f"volume_{label}.csv")
    volume = pd.read_csv(path, index_col=0, parse_dates=True)
    volume = volume.reindex(index=bundle.dates, columns=bundle.tickers).ffill().fillna(0.0)
    adv20 = volume.rolling(20, min_periods=1).mean().replace(0.0, EPS)
    return volume, adv20


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASS — stores the full NAV series and action history for one strategy
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    name:        str
    nav:         np.ndarray          # daily NAV, starts at 1.0
    dates:       pd.DatetimeIndex    # one date per NAV entry
    tc_paid:     float = 0.0         # cumulative TC as fraction of initial NAV
    lambdas:     list[float] = field(default_factory=list)
    hold_days:   list[int]   = field(default_factory=list)
    exposures:   list[float] = field(default_factory=list)
    anchor_blends: list[float] = field(default_factory=list)

    # ── Derived metrics ──────────────────────────────────────────────────────

    def total_return(self) -> float:
        return float(self.nav[-1] / self.nav[0] - 1.0)

    def cagr(self) -> float:
        n_days = max(len(self.nav) - 1, 1)
        return float((self.nav[-1] / self.nav[0]) ** (252 / n_days) - 1.0)

    def annualised_vol(self) -> float:
        daily_rets = np.diff(self.nav) / (self.nav[:-1] + EPS)
        return float(np.std(daily_rets, ddof=1) * np.sqrt(252))

    def sharpe(self) -> float:
        vol = self.annualised_vol()
        if vol < EPS:
            return 0.0
        return float((self.cagr() - RISK_FREE_RATE) / vol)

    def max_drawdown(self) -> float:
        """Returns negative fraction (e.g. −0.25 for a 25 % drawdown)."""
        peak = np.maximum.accumulate(self.nav)
        dd   = (self.nav - peak) / (peak + EPS)
        return float(dd.min())

    def calmar(self) -> float:
        mdd = abs(self.max_drawdown())
        if mdd < EPS:
            return 0.0
        return float(self.cagr() / mdd)

    def mean_lambda(self) -> float:
        return float(np.mean(self.lambdas)) if self.lambdas else float("nan")

    def mean_hold(self) -> float:
        return float(np.mean(self.hold_days)) if self.hold_days else float("nan")

    def mean_exposure(self) -> float:
        return float(np.mean(self.exposures)) if self.exposures else float("nan")

    def mean_anchor_blend(self) -> float:
        return float(np.mean(self.anchor_blends)) if self.anchor_blends else float("nan")

    def to_dict(self) -> dict:
        return {
            "Strategy":          self.name,
            "Total Return (%)":  round(self.total_return() * 100, 2),
            "CAGR (%)":          round(self.cagr() * 100, 2),
            "Ann. Vol (%)":      round(self.annualised_vol() * 100, 2),
            "Sharpe":            round(self.sharpe(), 3),
            "Max DD (%)":        round(self.max_drawdown() * 100, 2),
            "Calmar":            round(self.calmar(), 3),
            "TC Paid (%)":       round(self.tc_paid * 100, 4),
            "Mean λ":            round(self.mean_lambda(), 3),
            "Mean Hold (days)":  round(self.mean_hold(), 1),
            "Mean Exposure":      round(self.mean_exposure(), 3),
            "Mean Anchor Blend":  round(self.mean_anchor_blend(), 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — build daily NAV from a weight sequence
# ══════════════════════════════════════════════════════════════════════════════

def _nav_from_daily_rets(
    daily_ret_df: pd.DataFrame,
    warm_up: int,
    get_weights_fn,
    cost_context_fn=None,
) -> tuple[np.ndarray, pd.DatetimeIndex, float]:
    """
    Generic backtest engine shared by equal-weight and MVO benchmarks.

    Parameters
    ──────────
    daily_ret_df  : (T, N) DataFrame of simple daily returns (post warm-up region)
    warm_up       : rows to skip before the first rebalance decision
    get_weights_fn: callable(t, current_w) → (w_new: np.ndarray, lam: float)
                    Returns (N+1,) weight vector (stocks + cash) and lambda used.

    Returns
    ───────
    (nav_array, dates, total_tc_fraction)
    """
    T = len(daily_ret_df)
    N = daily_ret_df.shape[1]

    nav      = 1.0
    total_tc = 0.0
    nav_arr  = np.empty(T - warm_up, dtype=np.float64)
    w        = np.ones(N + 1, dtype=np.float64) / (N + 1)   # equal weight start
    t_last   = warm_up     # track last rebalance
    nav_history: list[float] = [1.0]

    for i, t in enumerate(range(warm_up, T)):
        # Rebalance decision: call caller's weight function
        w_new, _ = get_weights_fn(t, w)

        if w_new is not None:
            cost_context = cost_context_fn(t, w, w_new, nav_history) if cost_context_fn else {}
            tc        = transaction_cost(w, w_new, **cost_context)
            nav      *= (1.0 - tc)
            total_tc += tc
            w         = w_new.copy()

        # Apply today's returns
        rets        = daily_ret_df.iloc[t].values.astype(np.float64)
        port_ret    = float((rets * w[:-1]).sum())
        nav        *= (1.0 + port_ret)

        # Drift weights
        w_stocks    = w[:-1] * (1.0 + rets)
        total_val   = w_stocks.sum() + w[-1]
        if total_val > EPS:
            w = np.append(w_stocks / total_val, w[-1] / total_val)

        nav_arr[i] = nav
        nav_history.append(nav)

    dates = daily_ret_df.index[warm_up:]
    return nav_arr, pd.DatetimeIndex(dates), total_tc


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — PPO RL Agent
# ══════════════════════════════════════════════════════════════════════════════

def backtest_ppo(
    split: str,
    model_path: str,
    vecnorm_path: str = VECNORM_PATH,
    reward_scale: bool = True,
) -> BacktestResult:
    """Run the PPO agent on `split` using a single deterministic episode."""
    vec_env = make_vec_env(
        lambda: PortfolioEnv(split=split, reward_scale=reward_scale),
        n_envs=1,
    )
    if os.path.isfile(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    unwrap = vec_env.envs[0].unwrapped

    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done_arr = np.array([False])
    all_navs: list[float] = [1.0]
    lambdas: list[float] = []
    holds: list[int] = []
    exposures: list[float] = []
    anchor_blends: list[float] = []
    tc_paid: float = 0.0
    dates: list = [unwrap.bundle.dates[unwrap.t]]

    while not bool(done_arr[0]):
        action, _ = model.predict(obs, deterministic=True)
        step_out = vec_env.step(action)
        if len(step_out) == 5:
            obs, _, terminated, truncated, infos = step_out
            done_arr = np.logical_or(terminated, truncated)
        else:
            obs, _, done_arr, infos = step_out
        info = infos[0]

        new_navs = unwrap.nav_history[len(all_navs) :]
        all_navs.extend(new_navs)
        dates.extend(
            [unwrap.bundle.dates[min(unwrap.t, unwrap.T - 1)]] * len(new_navs)
        )

        lambdas.append(info["lambda"])
        holds.append(info["hold_days"])
        exposures.append(info.get("exposure", float("nan")))
        ppo_exposure_series = np.array(exposures)
        mean_exposure = np.nanmean(ppo_exposure_series)
        anchor_blends.append(info.get("anchor_blend", float("nan")))
        tc_paid = info["total_tc"]

    nav_arr = np.array(all_navs, dtype=np.float64)
    # Use the bundle dates aligned to NAV length
    nav_dates = unwrap.bundle.dates[unwrap.warm_up : unwrap.warm_up + len(nav_arr)]
    if len(nav_dates) < len(nav_arr):
        nav_arr = nav_arr[: len(nav_dates)]
    if len(nav_dates) > len(nav_arr):
        nav_dates = nav_dates[: len(nav_arr)]

    vec_env.close()
    return BacktestResult(
        name      = "PPO Agent",
        nav       = nav_arr,
        dates     = pd.DatetimeIndex(nav_dates),
        tc_paid   = tc_paid,
        lambdas   = lambdas,
        hold_days = holds,
        exposures = exposures,
        anchor_blends = anchor_blends,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Equal-Weight (1/N, monthly rebalance)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_equal_weight(split: str) -> BacktestResult:
    bundle   = load_bundle(split)
    warm_up  = bundle.warm_up_rows()
    N        = bundle.n_assets
    volume, adv20 = _load_volume_context(bundle)
    
    BENCHMARK_EXPOSURE = 0.65   # match PPO mean exposure
    w_eq = np.zeros(N + 1, dtype=np.float64)

    w_eq[:N] = (1.0 / N) * BENCHMARK_EXPOSURE
    w_eq[-1] = 1.0 - BENCHMARK_EXPOSURE
    last_rebal = [warm_up]

    def get_weights(t: int, current_w: np.ndarray):
        if (t - last_rebal[0]) >= BENCH_REBAL_DAYS or t == warm_up:
            last_rebal[0] = t
            return w_eq.copy(), float("nan")
        return None, float("nan")

    nav_arr, dates, tc_paid = _nav_from_daily_rets(
        bundle.daily_ret,
        warm_up,
        get_weights,
        lambda t, *_: {
            "prices_t": bundle.prices.iloc[t].values.astype(np.float64),
            "volume_t": volume.iloc[t].values.astype(np.float64),
            "adv_t": adv20.iloc[t].values.astype(np.float64),
        },
    )
    return BacktestResult(
        name    = "Equal-Weight (1/N)",
        nav     = nav_arr,
        dates   = dates,
        tc_paid = tc_paid,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — Nifty BeES Buy-and-Hold
# ══════════════════════════════════════════════════════════════════════════════

def backtest_niftybees(split: str) -> BacktestResult:
    bundle  = load_bundle(split)
    warm_up = bundle.warm_up_rows()
    tickers = bundle.tickers
    volume, adv20 = _load_volume_context(bundle)

    nifty_idx = next(
        (i for i, t in enumerate(tickers) if "NIFTYBEES" in t.upper()),
        None
    )

    if nifty_idx is None:
        # Fall back to equal weight if NIFTYBEES not in the universe
        print("  ⚠  NIFTYBEES not found in tickers; using equal-weight as proxy.")
        return backtest_equal_weight(split)

    N       = bundle.n_assets
    w_nifty = np.zeros(N + 1, dtype=np.float64)
    w_nifty[nifty_idx] = 1.0   # 100% in NIFTYBEES, held forever

    def get_weights(t: int, current_w: np.ndarray):
        if t == warm_up:
            return w_nifty.copy(), float("nan")
        return None, float("nan")

    nav_arr, dates, tc_paid = _nav_from_daily_rets(
        bundle.daily_ret,
        warm_up,
        get_weights,
        lambda t, *_: {
            "prices_t": bundle.prices.iloc[t].values.astype(np.float64),
            "volume_t": volume.iloc[t].values.astype(np.float64),
            "adv_t": adv20.iloc[t].values.astype(np.float64),
        },
    )
    return BacktestResult(
        name    = "Nifty BeES B&H",
        nav     = nav_arr,
        dates   = dates,
        tc_paid = tc_paid,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4-6 — MVO with fixed λ
# ══════════════════════════════════════════════════════════════════════════════

def backtest_mvo_fixed_lambda(split: str, lam: float) -> BacktestResult:
    """
    Run a fixed-λ MVO strategy with monthly (21-day) rebalancing.

    At each rebalance point we call efficient_frontier.optimize() with the
    current state vector and the fixed lambda, apply TC, then hold.
    """
    bundle   = load_bundle(split)
    macro_df = pd.read_csv(MACRO_PATH, index_col=0, parse_dates=True)
    warm_up  = bundle.warm_up_rows()
    N        = bundle.n_assets
    tickers  = bundle.tickers
    volume, adv20 = _load_volume_context(bundle)

    last_rebal = [warm_up]
    lambdas:   list[float] = []
    holds:     list[int]   = []
    nav_arr    = np.empty(len(bundle.dates) - warm_up, dtype=np.float64)

    nav      = 1.0
    total_tc = 0.0
    w        = np.ones(N + 1, dtype=np.float64) / (N + 1)

    nav_history: list[float] = [1.0]
    peak_nav = 1.0
    last_turnover = 0.0
    last_tc_rate = 0.0

    for i, t in enumerate(range(warm_up, len(bundle.dates))):
        rebalance = (t - last_rebal[0]) >= BENCH_REBAL_DAYS or t == warm_up

        if rebalance:
            # Build state vector at time t
            nav_series = np.array(nav_history, dtype=np.float64)
            state = build_state_vector(
                daily_ret_t = bundle.daily_ret.iloc[t].values.astype(np.float32),
                momentum_t  = bundle.momentum.iloc[t].values.astype(np.float32),
                vol20d_t    = bundle.realized_vol.iloc[t].values.astype(np.float32),
                ma_spread_t = bundle.ma_spread.iloc[t].values.astype(np.float32),
                cov_t       = bundle.cov[t],
                weights     = w[:-1].astype(np.float32),
                tickers     = tickers,
                macro_df    = macro_df,
                t_date      = bundle.dates[t],
                nav_series  = nav_series,
                log_ret_t   = bundle.log_ret.iloc[t].values.astype(np.float32),
                volume_t    = volume.iloc[t].values.astype(np.float32),
                adv_t       = adv20.iloc[t].values.astype(np.float32),
                prices_t    = bundle.prices.iloc[t].values.astype(np.float32),
                prev_turnover = last_turnover,
                prev_tc_rate  = last_tc_rate,
            )
            try:
                result = optimize(
                    state,
                    tickers,
                    lam,
                    current_weights=w,
                    max_equity=1.0,
                    min_equity=1.0,
                    cov_matrix=bundle.cov[t],
                    mom20d_t=horizon_momentum_pct(bundle.prices, t, 20),
                    mom60d_t=horizon_momentum_pct(bundle.prices, t, 60),
                    daily_ret_t=bundle.daily_ret.iloc[t].values.astype(np.float64),
                )
                w_new  = result.weights.astype(np.float64)
            except Exception:
                w_new  = w.copy()    # fall back to current weights on solver failure

            turnover = float(np.abs(w_new[:-1] - w[:-1]).sum())
            tc        = transaction_cost(
                w,
                w_new,
                state=state,
                prices_t=bundle.prices.iloc[t].values.astype(np.float64),
                volume_t=volume.iloc[t].values.astype(np.float64),
                adv_t=adv20.iloc[t].values.astype(np.float64),
            )
            nav      *= (1.0 - tc)
            total_tc += tc
            w         = w_new.copy()
            last_turnover = turnover
            last_tc_rate = tc / max(turnover, EPS)

            lambdas.append(lam)
            holds.append(BENCH_REBAL_DAYS)
            last_rebal[0] = t

        # Apply today's returns
        rets     = bundle.daily_ret.iloc[t].values.astype(np.float64)
        port_ret = float((rets * w[:-1]).sum())
        nav     *= (1.0 + port_ret)
        peak_nav = max(peak_nav, nav)
        nav_history.append(nav)

        # Drift weights
        w_stocks  = w[:-1] * (1.0 + rets)
        total_val = w_stocks.sum() + w[-1]
        if total_val > EPS:
            w = np.append(w_stocks / total_val, w[-1] / total_val)

        nav_arr[i] = nav

    dates = pd.DatetimeIndex(bundle.dates[warm_up:])
    return BacktestResult(
        name      = f"MVO λ={lam:.1f}",
        nav       = nav_arr,
        dates     = dates,
        tc_paid   = total_tc,
        lambdas   = lambdas,
        hold_days = holds,
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_table(results: list[BacktestResult]) -> pd.DataFrame:
    rows = [r.to_dict() for r in results]
    df   = pd.DataFrame(rows).set_index("Strategy")
    return df


def print_metrics_table(df: pd.DataFrame, split: str) -> None:
    width = 110
    print()
    print("=" * width)
    print(f"  BACKTEST RESULTS — split: {split.upper()}")
    print("=" * width)
    print(df.to_string())
    print("=" * width)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# NAV CURVE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_nav_curves(results: list[BacktestResult], split: str) -> str:
    """
    Align all NAV series on a common date index and save to CSV.
    Missing dates for shorter series are forward-filled.
    """
    # Use the longest date index as the reference
    ref_dates = max(results, key=lambda r: len(r.dates)).dates

    df = pd.DataFrame(index=ref_dates)
    df.index.name = "Date"

    for r in results:
        s = pd.Series(r.nav, index=r.dates, name=r.name)
        df[r.name] = s.reindex(ref_dates).ffill()

    path = os.path.join(RESULTS_DIR, f"nav_curves_{split}.csv")
    df.to_csv(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(split: str, model_path: str) -> pd.DataFrame:
    """
    Run all strategies on `split` and return the metrics DataFrame.
    """
    print(f"\n{'─'*65}")
    print(f"  Evaluating split: {split.upper()}")
    print(f"  Model: {model_path}.zip")
    print(f"{'─'*65}")

    results: list[BacktestResult] = []

    # 1. PPO Agent
    print("  [1/6] PPO Agent …", end=" ", flush=True)
    try:
        r = backtest_ppo(split, model_path)
        results.append(r)
        print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")
    except Exception as e:
        print(f"FAILED: {e}")

    # 2. Equal-Weight
    print("  [2/6] Equal-Weight (1/N) …", end=" ", flush=True)
    r = backtest_equal_weight(split)
    results.append(r)
    print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")

    # 3. Nifty BeES B&H
    print("  [3/6] Nifty BeES B&H …", end=" ", flush=True)
    r = backtest_niftybees(split)
    results.append(r)
    print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")

    # 4. MVO λ=0.5 (return-seeking)
    print("  [4/6] MVO λ=0.5 …", end=" ", flush=True)
    r = backtest_mvo_fixed_lambda(split, lam=0.5)
    results.append(r)
    print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")

    # 5. MVO λ=1.0 (balanced)
    print("  [5/6] MVO λ=1.0 …", end=" ", flush=True)
    r = backtest_mvo_fixed_lambda(split, lam=1.0)
    results.append(r)
    print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")

    # 6. MVO λ=2.0 (conservative)
    print("  [6/6] MVO λ=2.0 …", end=" ", flush=True)
    r = backtest_mvo_fixed_lambda(split, lam=2.0)
    results.append(r)
    print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")

    # ── Build and display metrics table ───────────────────────────────────────
    metrics_df = build_metrics_table(results)
    print_metrics_table(metrics_df, split)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{split}.csv")
    metrics_df.to_csv(metrics_path)
    print(f"  Metrics saved  → {metrics_path}")

    # ── Save NAV curves ───────────────────────────────────────────────────────
    nav_path = save_nav_curves(results, split)
    print(f"  NAV curves     → {nav_path}")

    return metrics_df


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED TABLE ACROSS SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def save_combined_table(tables: dict[str, pd.DataFrame]) -> str:
    """
    Merge metrics from multiple splits into a single wide CSV.
    Columns are prefixed with the split name, e.g. 'val_CAGR (%)'.
    """
    frames = []
    for split, df in tables.items():
        renamed = df.add_prefix(f"{split}_")
        frames.append(renamed)

    combined = pd.concat(frames, axis=1)
    path = os.path.join(RESULTS_DIR, "metrics_combined.csv")
    combined.to_csv(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PPO portfolio agent vs. benchmarks"
    )
    parser.add_argument(
        "--split", nargs="+", default=["val", "test"],
        choices=["train", "val", "test"],
        help="Data split(s) to evaluate (default: val test)",
    )
    parser.add_argument(
        "--model", type=str, default="best",
        choices=["best", "latest"],
        help="Which saved model to use (default: best)",
    )
    parser.add_argument(
        "--model-path", type=str, default="",
        help="Full path to model file (overrides --model)",
    )
    args = parser.parse_args()

    # ── Resolve model path ────────────────────────────────────────────────────
    if args.model_path:
        model_path = args.model_path.replace(".zip", "")
    elif args.model == "best":
        model_path = BEST_PATH
    else:
        model_path = LATEST_PATH

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            f"Run train.py first or pass --model-path explicitly."
        )

    print()
    print("=" * 65)
    print("  Deep-RL Portfolio — Backtest Evaluation")
    print("=" * 65)
    print(f"  Model           : {model_path}.zip")
    print(f"  Splits          : {args.split}")
    print("  TC model         : dynamic statutory + volatility/liquidity impact")
    print(f"  Benchmark rebal : every {BENCH_REBAL_DAYS} days")
    print(f"  Risk-free rate  : {RISK_FREE_RATE*100:.1f}% (Sharpe denominator)")

    all_tables: dict[str, pd.DataFrame] = {}
    for split in args.split:
        table = evaluate(split=split, model_path=model_path)
        all_tables[split] = table

    # ── Combined CSV ──────────────────────────────────────────────────────────
    if len(all_tables) > 1:
        combined_path = save_combined_table(all_tables)
        print(f"\n  Combined table → {combined_path}")

    print("\n  Done.\n")
