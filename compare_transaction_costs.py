"""
compare_transaction_costs.py — compare dynamic and static transaction costs.

This script keeps the trading logic fixed and changes only the transaction-cost
function used by env.py/evaluate.py:

  dynamic: state_spec.transaction_cost()
           statutory delivery charges + volatility/liquidity/regime impact

  static : TRANSACTION_COST_RATE * stock turnover
           legacy flat-cost model, currently 0.1% * turnover

Outputs
-------
  results/metrics_dynamic_<split>.csv
  results/metrics_static_<split>.csv
  results/tc_compare_<split>.csv

Usage
-----
  python3.13 compare_transaction_costs.py --split test
  python3.13 compare_transaction_costs.py --split val test
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

import env
import evaluate
from state_spec import TRANSACTION_COST_RATE, transaction_cost as dynamic_transaction_cost


RESULTS_DIR = "results"
MODEL_DIR = "models"
BEST_PATH = os.path.join(MODEL_DIR, "best_model")
LATEST_PATH = os.path.join(MODEL_DIR, "ppo_portfolio_latest")


def static_transaction_cost(
    w_old: np.ndarray,
    w_new: np.ndarray,
    **_: object,
) -> float:
    """Legacy static cost: flat 0.1% applied to stock turnover only."""
    turnover = float(np.abs(np.asarray(w_new[:-1]) - np.asarray(w_old[:-1])).sum())
    return TRANSACTION_COST_RATE * turnover


def _set_cost_model(cost_fn: Callable[..., float]) -> None:
    """Patch both modules because each imports transaction_cost into its globals."""
    env.transaction_cost = cost_fn
    evaluate.transaction_cost = cost_fn


def _run_strategy_set(split: str, model_path: str) -> pd.DataFrame:
    """Run the same strategy set used in evaluate.py without overwriting files."""
    results = []
    results.append(evaluate.backtest_ppo(split, model_path))
    results.append(evaluate.backtest_equal_weight(split))
    results.append(evaluate.backtest_niftybees(split))
    results.append(evaluate.backtest_mvo_fixed_lambda(split, lam=0.5))
    results.append(evaluate.backtest_mvo_fixed_lambda(split, lam=1.0))
    results.append(evaluate.backtest_mvo_fixed_lambda(split, lam=2.0))
    return evaluate.build_metrics_table(results)


def compare_split(split: str, model_path: str) -> pd.DataFrame:
    """Evaluate dynamic vs static transaction costs for one split."""
    print(f"\n{'=' * 90}")
    print(f"  Transaction Cost Comparison — {split.upper()}")
    print(f"{'=' * 90}")

    print("  Running dynamic TC model …", flush=True)
    _set_cost_model(dynamic_transaction_cost)
    dynamic_df = _run_strategy_set(split, model_path)

    print("  Running static TC model …", flush=True)
    _set_cost_model(static_transaction_cost)
    static_df = _run_strategy_set(split, model_path)

    # Restore dynamic model for any later imports/interactive use.
    _set_cost_model(dynamic_transaction_cost)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dynamic_path = os.path.join(RESULTS_DIR, f"metrics_dynamic_{split}.csv")
    static_path = os.path.join(RESULTS_DIR, f"metrics_static_{split}.csv")
    dynamic_df.to_csv(dynamic_path)
    static_df.to_csv(static_path)

    comparison = pd.DataFrame(index=dynamic_df.index)
    comparison["Static Return (%)"] = static_df["Total Return (%)"]
    comparison["Dynamic Return (%)"] = dynamic_df["Total Return (%)"]
    comparison["Return Δ dyn-static (pp)"] = (
        dynamic_df["Total Return (%)"] - static_df["Total Return (%)"]
    ).round(2)
    comparison["Static TC Paid (%)"] = static_df["TC Paid (%)"]
    comparison["Dynamic TC Paid (%)"] = dynamic_df["TC Paid (%)"]
    comparison["TC Δ dyn-static (pp)"] = (
        dynamic_df["TC Paid (%)"] - static_df["TC Paid (%)"]
    ).round(4)
    comparison["Static Sharpe"] = static_df["Sharpe"]
    comparison["Dynamic Sharpe"] = dynamic_df["Sharpe"]
    comparison["Sharpe Δ"] = (dynamic_df["Sharpe"] - static_df["Sharpe"]).round(3)
    comparison["Static Max DD (%)"] = static_df["Max DD (%)"]
    comparison["Dynamic Max DD (%)"] = dynamic_df["Max DD (%)"]

    compare_path = os.path.join(RESULTS_DIR, f"tc_compare_{split}.csv")
    comparison.to_csv(compare_path)

    print()
    print(comparison.to_string())
    print()
    print(f"  Dynamic metrics → {dynamic_path}")
    print(f"  Static metrics  → {static_path}")
    print(f"  Comparison      → {compare_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare dynamic transaction costs against static flat costs."
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Data split(s) to compare (default: test).",
    )
    parser.add_argument(
        "--model",
        choices=["best", "latest"],
        default="best",
        help="Saved PPO model to evaluate (default: best).",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Explicit model path without .zip; overrides --model.",
    )
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path.replace(".zip", "")
    elif args.model == "best":
        model_path = BEST_PATH
    else:
        model_path = LATEST_PATH

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    print("Deep-RL Portfolio — Transaction Cost Model Test")
    print(f"Model: {model_path}.zip")
    print(f"Static rate: {TRANSACTION_COST_RATE * 100:.3f}% x turnover")
    print("Dynamic model: statutory delivery charges + volatility/liquidity impact")

    for split_name in args.split:
        compare_split(split_name, model_path)
