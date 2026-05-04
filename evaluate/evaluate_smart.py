"""
evaluate_smart.py — Smart Tangency Portfolio Backtest & Benchmark Comparison
=============================================================================

Evaluates all 6 DRL agents + ensemble + static tangency benchmarks, as
described in Yu & Chang (2025).

Strategies
──────────
  DRL agents (6):
    PPO-MV    PPO-SV    PPO-CVaR
    A2C-MV    A2C-SV    A2C-CVaR

  Ensemble (1):
    Best agent by median Sharpe on training tail (see train_smart.py)
    NOTE: uses train-tail selection, NOT val split, to avoid temporal leakage

  Static benchmarks (5):
    Equal-Weight (1/N, monthly rebalance)
    Static Tangency MV    — max-Sharpe MVO, monthly rebalance
    Static Tangency SV    — max-Sharpe Semivariance, monthly rebalance
    Static Tangency CVaR  — max-Sharpe CVaR, monthly rebalance
    Buy-and-Hold baseline (equal-weight, held from start)

Metrics
───────
  Total Return (%)  ·  CAGR (%)  ·  Ann. Vol (%)
  Sharpe Ratio  ·  Max Drawdown (%)  ·  Calmar Ratio
  TC Paid (%)  ·  Mean λ idx  ·  Mean Hold (days)

Usage
─────
  python3.13 evaluate_smart.py                  # val + test, ensemble model
  python3.13 evaluate_smart.py --split test     # test only
  python3.13 evaluate_smart.py --no-drl         # benchmarks only (fast)
  python3.13 evaluate_smart.py --utility mv     # DRL agents for MV only
  python3.13 evaluate_smart.py --regimes        # also run bear/bull/sideways sub-periods
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C

from envs.env_smart import SmartPortfolioEnv, HOLDING_PERIODS_SMART
from core.smart_tangency import (
    SmartOptimizer,
    extract_smart_inputs,
    find_tangency,
    UTILITY_MV, UTILITY_SV, UTILITY_CVAR,
    UTILITIES,
    N_LAMBDA,
)
from features import load_bundle
from state_spec import (
    STATE,
    build_state_vector,
    transaction_cost,
    TRANSACTION_COST_RATE,
    MACRO_PATH,
    EPS,
)

RESULTS_DIR = "results"
MODEL_DIR   = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)

BENCH_REBAL_DAYS = 21       # monthly rebalance for static strategies
RISK_FREE_RATE   = 0.065    # India 1-yr T-bill (approx)
ALGOS            = ["ppo", "a2c"]

# ── Market regime sub-periods (within the test/val splits) ────────────────────
# These let us check whether DRL adds value in drawdown periods, not just
# bull markets.  Defined as date slices applied on top of whatever split is run.
# Identified from the Nifty50 price series:
#   Bull  : Jan 2021 – Oct 2021  (strong post-COVID recovery)
#   Bear  : Jan 2022 – Jun 2022  (rate-hike sell-off)
#   Sideways: Jul 2022 – Dec 2022 (choppy recovery)
REGIMES: dict[str, tuple[str, str]] = {
    "bull":     ("2021-01-01", "2021-10-31"),
    "bear":     ("2022-01-01", "2022-06-30"),
    "sideways": ("2022-07-01", "2022-12-31"),
}


# ══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    name:       str
    nav:        np.ndarray
    dates:      pd.DatetimeIndex
    tc_paid:    float = 0.0
    lambdas:    list[int]   = field(default_factory=list)
    hold_days:  list[int]   = field(default_factory=list)

    def total_return(self) -> float:
        return float(self.nav[-1] / self.nav[0] - 1.0)

    def cagr(self) -> float:
        n = max(len(self.nav) - 1, 1)
        return float((self.nav[-1] / self.nav[0]) ** (252 / n) - 1.0)

    def annualised_vol(self) -> float:
        dr = np.diff(self.nav) / (self.nav[:-1] + EPS)
        return float(np.std(dr, ddof=1) * np.sqrt(252))

    def sharpe(self) -> float:
        v = self.annualised_vol()
        return float((self.cagr() - RISK_FREE_RATE) / v) if v > EPS else 0.0

    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.nav)
        return float(((self.nav - peak) / (peak + EPS)).min())

    def calmar(self) -> float:
        mdd = abs(self.max_drawdown())
        return float(self.cagr() / mdd) if mdd > EPS else 0.0

    def mean_lambda(self) -> float:
        return float(np.mean(self.lambdas)) if self.lambdas else float("nan")

    def mean_hold(self) -> float:
        return float(np.mean(self.hold_days)) if self.hold_days else float("nan")

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
            "Mean λ idx":        round(self.mean_lambda(), 1),
            "Mean Hold (days)":  round(self.mean_hold(), 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY: DRL AGENT (PPO or A2C)
# ══════════════════════════════════════════════════════════════════════════════

def _model_path(algo: str, utility: str) -> str:
    candidates = [
        os.path.join(MODEL_DIR, f"smart_{algo}_{utility}_latest"),
        os.path.join(MODEL_DIR, f"smart_{algo}_{utility}_best"),
        os.path.join(MODEL_DIR, "best_model"),   # original env best
    ]
    for p in candidates:
        if os.path.exists(p + ".zip"):
            return p
    return candidates[0]   # will fail gracefully in backtest_drl


def backtest_drl(algo: str, utility: str, split: str) -> BacktestResult | None:
    """Run one DRL agent episode deterministically on `split`."""
    path = _model_path(algo, utility)
    if not os.path.exists(path + ".zip"):
        return None

    AlgoCls = PPO if algo == "ppo" else A2C
    try:
        model = AlgoCls.load(path)
    except Exception as e:
        print(f"    ✗ Load failed: {e}")
        return None

    env  = SmartPortfolioEnv(split=split, utility=utility)
    obs, _ = env.reset()

    all_navs: list[float] = [1.0]
    lambdas:  list[int]   = []
    holds:    list[int]   = []
    tc_paid:  float       = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_navs = env.nav_history[len(all_navs):]
        all_navs.extend(new_navs)
        lambdas.append(info["lambda_idx"])
        holds.append(info["hold_days"])
        tc_paid = info["total_tc"]

    nav_arr   = np.array(all_navs, dtype=np.float64)
    nav_dates = env.bundle.dates[env.warm_up: env.warm_up + len(nav_arr)]
    if len(nav_dates) < len(nav_arr):
        nav_arr = nav_arr[:len(nav_dates)]
    if len(nav_dates) > len(nav_arr):
        nav_dates = nav_dates[:len(nav_arr)]

    env.close()
    return BacktestResult(
        name      = f"{algo.upper()}-{utility.upper()}",
        nav       = nav_arr,
        dates     = pd.DatetimeIndex(nav_dates),
        tc_paid   = tc_paid,
        lambdas   = lambdas,
        hold_days = holds,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY: ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def backtest_ensemble(split: str) -> BacktestResult | None:
    """Load and run the ensemble (best-validation) model."""
    try:
        from train.train_smart import load_ensemble_model
        model, utility = load_ensemble_model()
    except Exception as e:
        print(f"    ✗ Ensemble load failed: {e}")
        return None

    algo = "ppo"  # default; load_ensemble_model may return either
    env  = SmartPortfolioEnv(split=split, utility=utility)
    obs, _ = env.reset()

    all_navs: list[float] = [1.0]
    lambdas:  list[int]   = []
    holds:    list[int]   = []
    tc_paid   = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_navs = env.nav_history[len(all_navs):]
        all_navs.extend(new_navs)
        lambdas.append(info["lambda_idx"])
        holds.append(info["hold_days"])
        tc_paid = info["total_tc"]

    nav_arr   = np.array(all_navs, dtype=np.float64)
    nav_dates = env.bundle.dates[env.warm_up: env.warm_up + len(nav_arr)]
    if len(nav_dates) < len(nav_arr):
        nav_arr = nav_arr[:len(nav_dates)]
    if len(nav_dates) > len(nav_arr):
        nav_dates = nav_dates[:len(nav_arr)]

    env.close()
    return BacktestResult(
        name      = f"Ensemble ({utility.upper()})",
        nav       = nav_arr,
        dates     = pd.DatetimeIndex(nav_dates),
        tc_paid   = tc_paid,
        lambdas   = lambdas,
        hold_days = holds,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY: EQUAL-WEIGHT
# ══════════════════════════════════════════════════════════════════════════════

def backtest_equal_weight(split: str) -> BacktestResult:
    bundle  = load_bundle(split)
    warm_up = bundle.warm_up_rows()
    N       = bundle.n_assets
    T       = len(bundle.dates)

    w_eq = np.zeros(N + 1, dtype=np.float64)
    w_eq[:N] = 1.0 / N

    nav = 1.0; total_tc = 0.0
    nav_arr  = np.empty(T - warm_up, dtype=np.float64)
    w        = w_eq.copy()
    last_rb  = warm_up

    for i, t in enumerate(range(warm_up, T)):
        if (t - last_rb) >= BENCH_REBAL_DAYS or t == warm_up:
            tc = float(TRANSACTION_COST_RATE * np.abs(w_eq[:N] - w[:N]).sum())
            nav *= (1.0 - tc); total_tc += tc
            w = w_eq.copy(); last_rb = t

        rets     = bundle.daily_ret.iloc[t].values.astype(np.float64)
        port_ret = float((rets * w[:N]).sum())
        nav     *= (1.0 + port_ret)

        ws = w[:N] * (1.0 + rets)
        s  = ws.sum() + w[-1]
        if s > EPS:
            w = np.append(ws / s, w[-1] / s)
        nav_arr[i] = nav

    dates = pd.DatetimeIndex(bundle.dates[warm_up:])
    return BacktestResult("Equal-Weight (1/N)", nav_arr, dates, total_tc)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY: STATIC TANGENCY (max-Sharpe, monthly rebalance)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_static_tangency(split: str, utility: str) -> BacktestResult:
    """
    Static tangency portfolio: scan the frontier each rebalance date and
    pick the max-Sharpe allocation, then hold for BENCH_REBAL_DAYS.
    This is the paper's static tangency benchmark.
    """
    bundle   = load_bundle(split)
    macro_df = pd.read_csv(MACRO_PATH, index_col=0, parse_dates=True)
    warm_up  = bundle.warm_up_rows()
    N        = bundle.n_assets
    T        = len(bundle.dates)
    tickers  = bundle.tickers
    opt      = SmartOptimizer(utility)

    nav = 1.0; total_tc = 0.0
    nav_arr  = np.empty(T - warm_up, dtype=np.float64)
    w        = np.ones(N + 1) / (N + 1)
    last_rb  = warm_up
    nav_hist = [1.0]
    lambdas: list[int] = []
    holds:   list[int] = []

    for i, t in enumerate(range(warm_up, T)):
        rebalance = (t - last_rb) >= BENCH_REBAL_DAYS or t == warm_up

        if rebalance:
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
                nav_series  = np.array(nav_hist),
                log_ret_t   = bundle.log_ret.iloc[t].values.astype(np.float32),
            )
            inputs = extract_smart_inputs(state, bundle, t)
            try:
                tang = opt.tangency(
                    mu           = inputs["mu"],
                    cov          = inputs["cov"],
                    tickers      = tickers,
                    returns_hist = inputs["returns_hist"],
                )
                w_new = tang.weights.astype(np.float64)
                lambdas.append(tang.lam_idx)
            except Exception:
                w_new = w.copy()
                lambdas.append(N_LAMBDA // 2)

            tc        = float(TRANSACTION_COST_RATE * np.abs(w_new[:N] - w[:N]).sum())
            nav      *= (1.0 - tc); total_tc += tc
            w         = w_new.copy()
            holds.append(BENCH_REBAL_DAYS)
            last_rb   = t

        rets     = bundle.daily_ret.iloc[t].values.astype(np.float64)
        port_ret = float((rets * w[:N]).sum())
        nav     *= (1.0 + port_ret)
        nav_hist.append(nav)

        ws = w[:N] * (1.0 + rets)
        s  = ws.sum() + w[-1]
        if s > EPS:
            w = np.append(ws / s, w[-1] / s)
        nav_arr[i] = nav

    dates = pd.DatetimeIndex(bundle.dates[warm_up:])
    return BacktestResult(
        name      = f"Tangency-{utility.upper()} (static)",
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
    return pd.DataFrame(rows).set_index("Strategy")


def print_table(df: pd.DataFrame, split: str) -> None:
    w = 120
    print("\n" + "=" * w)
    print(f"  SMART TANGENCY BACKTEST — split: {split.upper()}")
    print("=" * w)
    print(df.to_string())
    print("=" * w)


def save_nav_curves(results: list[BacktestResult], split: str) -> str:
    ref_dates = max(results, key=lambda r: len(r.dates)).dates
    df = pd.DataFrame(index=ref_dates)
    df.index.name = "Date"
    for r in results:
        s = pd.Series(r.nav, index=r.dates, name=r.name)
        df[r.name] = s.reindex(ref_dates).ffill()
    path = os.path.join(RESULTS_DIR, f"nav_curves_smart_{split}.csv")
    df.to_csv(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(split:      str,
             run_drl:    bool      = True,
             utilities:  list[str] = list(UTILITIES),
             algos:      list[str] = ALGOS,
             run_regimes: bool     = False) -> pd.DataFrame:

    print(f"\n{'─'*65}")
    print(f"  Evaluating split: {split.upper()}")
    print(f"{'─'*65}")

    results: list[BacktestResult] = []
    step = [0]

    def _run(label: str, fn, *args, **kwargs):
        step[0] += 1
        print(f"  [{step[0]:>2}] {label} …", end=" ", flush=True)
        try:
            r = fn(*args, **kwargs)
            if r is not None:
                results.append(r)
                print(f"done  NAV={r.nav[-1]:.4f}  ret={r.total_return()*100:+.2f}%")
            else:
                print("SKIPPED (model not found)")
        except Exception as e:
            print(f"ERROR: {e}")

    # ── DRL agents ─────────────────────────────────────────────────────────────
    if run_drl:
        for algo, utility in product(algos, utilities):
            _run(f"{algo.upper()}-{utility.upper()}",
                 backtest_drl, algo, utility, split)

        _run("Ensemble", backtest_ensemble, split)

    # ── Static benchmarks ──────────────────────────────────────────────────────
    _run("Equal-Weight (1/N)", backtest_equal_weight, split)

    for utility in utilities:
        _run(f"Tangency-{utility.upper()} (static)",
             backtest_static_tangency, split, utility)

    if not results:
        print("  ✗ No results to display.")
        return pd.DataFrame()

    metrics_df = build_metrics_table(results)
    print_table(metrics_df, split)

    metrics_path = os.path.join(RESULTS_DIR, f"metrics_smart_{split}.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\n  Metrics saved  → {metrics_path}")

    nav_path = save_nav_curves(results, split)
    print(f"  NAV curves     → {nav_path}")

    if run_regimes:
        evaluate_regimes(results, split)

    return metrics_df


# ══════════════════════════════════════════════════════════════════════════════
# REGIME ANALYSIS — slice backtest results into market sub-periods
# ══════════════════════════════════════════════════════════════════════════════

def _slice_result(r: BacktestResult,
                  date_from: str,
                  date_to:   str,
                  label_suffix: str) -> BacktestResult | None:
    """
    Slice a BacktestResult to a date range and return a new BacktestResult.
    Returns None if the slice contains fewer than 10 data points.
    """
    mask = (r.dates >= date_from) & (r.dates <= date_to)
    if mask.sum() < 10:
        return None
    sliced_nav   = r.nav[mask]
    sliced_dates = r.dates[mask]
    # Re-normalise NAV to start at 1.0 in the sub-period
    sliced_nav   = sliced_nav / sliced_nav[0]
    return BacktestResult(
        name      = f"{r.name} [{label_suffix}]",
        nav       = sliced_nav,
        dates     = sliced_dates,
        tc_paid   = r.tc_paid,   # carry-over (approximation)
        lambdas   = r.lambdas,
        hold_days = r.hold_days,
    )


def evaluate_regimes(full_results: list[BacktestResult],
                     split:        str) -> None:
    """
    For each defined market regime sub-period, slice all full backtest results
    and print a compact metrics table showing regime-conditional performance.

    This answers the key question: does DRL add value in bear markets, or only
    in the 2021 bull run where static MV also performs well?
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  REGIME ANALYSIS — {split.upper()}")
    print(sep)

    for regime_name, (d_from, d_to) in REGIMES.items():
        sliced = []
        for r in full_results:
            s = _slice_result(r, d_from, d_to, regime_name)
            if s is not None:
                sliced.append(s)

        if not sliced:
            print(f"\n  [{regime_name.upper()}] No data in range {d_from} -> {d_to}")
            continue

        df = build_metrics_table(sliced)
        # Keep only the most informative columns for the compact regime view
        cols = ["Total Return (%)", "Sharpe", "Max DD (%)", "Calmar"]
        df_show = df[[c for c in cols if c in df.columns]]
        df_show = df_show.sort_values("Sharpe", ascending=False)

        print(f"\n  [{regime_name.upper()}]  {d_from}  ->  {d_to}")
        print(df_show.to_string())

        path = os.path.join(RESULTS_DIR,
                            f"metrics_smart_{split}_{regime_name}.csv")
        df.to_csv(path)
        print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Tangency Portfolio — Backtest Evaluation"
    )
    parser.add_argument("--split", nargs="+", default=["val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--utility", type=str, default="all",
                        choices=list(UTILITIES) + ["all"])
    parser.add_argument("--algo", type=str, default="all",
                        choices=ALGOS + ["all"])
    parser.add_argument("--no-drl", action="store_true",
                        help="Skip DRL agent backtests; run benchmarks only")
    parser.add_argument("--regimes", action="store_true",
                        help="Run regime sub-period analysis (bear/bull/sideways)")
    args = parser.parse_args()

    utilities = list(UTILITIES)  if args.utility == "all" else [args.utility]
    algos     = ALGOS            if args.algo    == "all" else [args.algo]

    print()
    print("=" * 65)
    print("  Smart Tangency Portfolio — Backtest Evaluation")
    print("=" * 65)
    print(f"  Splits     : {args.split}")
    print(f"  Utilities  : {utilities}")
    print(f"  Algorithms : {algos}")
    print(f"  TC rate    : {TRANSACTION_COST_RATE*100:.1f}%")

    all_tables: dict[str, pd.DataFrame] = {}
    for split in args.split:
        df = evaluate(
            split       = split,
            run_drl     = not args.no_drl,
            utilities   = utilities,
            algos       = algos,
            run_regimes = args.regimes,
        )
        all_tables[split] = df

    if len(all_tables) > 1:
        frames = []
        for sp, df in all_tables.items():
            frames.append(df.add_prefix(f"{sp}_"))
        combined = pd.concat(frames, axis=1)
        path = os.path.join(RESULTS_DIR, "metrics_smart_combined.csv")
        combined.to_csv(path)
        print(f"\n  Combined table → {path}")

    print("\n  Done.")
