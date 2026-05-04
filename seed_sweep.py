"""
seed_sweep.py — train/evaluate multiple PPO seeds with validation-only selection.

Protocol:
  1. Train each seed independently.
  2. Evaluate every seed on validation only.
  3. Select the winner by validation excess-return/risk score.
  4. Optionally run test once for the selected winner.

This avoids implicitly tuning on the test split.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd

import train as train_module
from evaluate import (
    RESULTS_DIR,
    BacktestResult,
    backtest_equal_weight,
    backtest_mvo_fixed_lambda,
    backtest_niftybees,
    backtest_ppo,
    build_metrics_table,
)


SEED_MODEL_DIR = os.path.join(train_module.MODEL_DIR, "seed_runs")
SWEEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "seed_sweeps")


@dataclass
class SeedArtifacts:
    seed: int
    model_path: str
    vecnorm_path: str


def _copy_if_exists(src: str, dst: str) -> bool:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def _save_seed_artifacts(seed: int) -> SeedArtifacts:
    """Persist the latest seed-specific model and VecNormalize stats."""
    os.makedirs(SEED_MODEL_DIR, exist_ok=True)
    prefix = os.path.join(SEED_MODEL_DIR, f"seed_{seed}")

    latest_model = f"{train_module.MODEL_PATH}.zip"
    best_validation_model = f"{train_module.BEST_VALIDATION_PATH}.zip"
    latest_vecnorm = train_module.VECNORM_PATH
    best_validation_vecnorm = os.path.join(train_module.MODEL_DIR, "vecnorm_best_validation.pkl")

    model_dst = f"{prefix}_best_validation.zip"
    vecnorm_dst = f"{prefix}_vecnorm.pkl"

    if not _copy_if_exists(best_validation_model, model_dst):
        _copy_if_exists(latest_model, model_dst)

    if not _copy_if_exists(best_validation_vecnorm, vecnorm_dst):
        _copy_if_exists(latest_vecnorm, vecnorm_dst)

    return SeedArtifacts(
        seed=seed,
        model_path=model_dst.replace(".zip", ""),
        vecnorm_path=vecnorm_dst,
    )


def _run_strategy_set(
    split: str,
    model_path: str,
    vecnorm_path: str,
) -> list[BacktestResult]:
    return [
        backtest_ppo(split, model_path, vecnorm_path=vecnorm_path),
        backtest_equal_weight(split),
        backtest_niftybees(split),
        backtest_mvo_fixed_lambda(split, lam=0.5),
        backtest_mvo_fixed_lambda(split, lam=1.0),
        backtest_mvo_fixed_lambda(split, lam=2.0),
    ]


def _score_vs_equal_weight(ppo: BacktestResult, equal_weight: BacktestResult) -> dict[str, float]:
    """Validation score aligned to after-cost excess return with risk controls."""
    n = min(len(ppo.nav), len(equal_weight.nav))
    ppo_nav = ppo.nav[:n]
    ew_nav = equal_weight.nav[:n]

    ppo_log = np.diff(np.log(np.maximum(ppo_nav, 1e-12)))
    ew_log = np.diff(np.log(np.maximum(ew_nav, 1e-12)))
    excess_log = ppo_log - ew_log

    ppo_ret = float(ppo_nav[-1] / ppo_nav[0] - 1.0)
    ew_ret = float(ew_nav[-1] / ew_nav[0] - 1.0)
    excess_ret = ppo_ret - ew_ret
    annual_median_excess = float(np.median(excess_log) * 252.0) if len(excess_log) else 0.0
    max_dd = abs(ppo.max_drawdown())
    score = annual_median_excess - ppo.tc_paid - 0.5 * max_dd

    return {
        "ppo_return": ppo_ret,
        "ew_return": ew_ret,
        "excess_return": excess_ret,
        "annual_median_excess_log": annual_median_excess,
        "ppo_sharpe": ppo.sharpe(),
        "ppo_max_dd": ppo.max_drawdown(),
        "ppo_tc_paid": ppo.tc_paid,
        "ppo_mean_exposure": ppo.mean_exposure(),
        "ppo_mean_anchor_blend": ppo.mean_anchor_blend(),
        "score": score,
    }


def _evaluate_seed(
    artifacts: SeedArtifacts,
    split: str,
    output_dir: str,
) -> dict[str, float]:
    results = _run_strategy_set(split, artifacts.model_path, artifacts.vecnorm_path)
    metrics = build_metrics_table(results)
    metrics_path = os.path.join(output_dir, f"metrics_{split}_seed_{artifacts.seed}.csv")
    metrics.to_csv(metrics_path)

    ppo = next(r for r in results if r.name == "PPO Agent")
    ew = next(r for r in results if r.name == "Equal-Weight (1/N)")
    row = _score_vs_equal_weight(ppo, ew)
    row.update({
        "seed": artifacts.seed,
        "split": split,
        "model_path": artifacts.model_path,
        "vecnorm_path": artifacts.vecnorm_path,
        "metrics_path": metrics_path,
    })
    return row


def _aggregate(rows: list[dict[str, float]], path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    numeric = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        "mean": numeric.mean(),
        "std": numeric.std(ddof=1),
        "min": numeric.min(),
        "max": numeric.max(),
    })
    summary.to_csv(path.replace(".csv", "_summary.csv"))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multiple seeds, select on validation, optionally run final test."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 21, 31, 41, 51])
    parser.add_argument("--timesteps", type=int, default=train_module.DEFAULT_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=train_module.DEFAULT_N_ENVS)
    parser.add_argument("--no-vecnorm", action="store_true")
    parser.add_argument("--no-reward-scale", action="store_true")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="After selecting by validation, evaluate the winning seed on test once.",
    )
    parser.add_argument(
        "--eval-test-all",
        action="store_true",
        help="Diagnostic only: evaluate all seeds on test. Do not use for model selection.",
    )
    args = parser.parse_args()

    os.makedirs(SWEEP_RESULTS_DIR, exist_ok=True)
    reward_scale = not args.no_reward_scale
    use_vecnorm = not args.no_vecnorm

    print("=" * 72)
    print("  PPO Multi-Seed Validation Sweep")
    print("=" * 72)
    print(f"  Seeds       : {args.seeds}")
    print(f"  Timesteps   : {args.timesteps:,}")
    print(f"  n_envs      : {args.n_envs}")
    print(f"  VecNormalize: {use_vecnorm}")
    print("  Selection   : validation score vs Equal-Weight")
    print()

    val_rows: list[dict[str, float]] = []
    artifacts_by_seed: dict[int, SeedArtifacts] = {}

    for seed in args.seeds:
        print(f"\n{'-' * 72}")
        print(f"  Training seed {seed}")
        print(f"{'-' * 72}")
        train_module.train(
            total_timesteps=args.timesteps,
            resume_path="",
            reward_scale=reward_scale,
            use_vecnorm=use_vecnorm,
            n_envs=args.n_envs,
            seed=seed,
        )
        artifacts = _save_seed_artifacts(seed)
        artifacts_by_seed[seed] = artifacts

        print(f"\n  Evaluating seed {seed} on validation only")
        row = _evaluate_seed(artifacts, "val", SWEEP_RESULTS_DIR)
        val_rows.append(row)
        print(
            f"  val score={row['score']:+.4f} "
            f"excess={row['excess_return']*100:+.2f}% "
            f"Sharpe={row['ppo_sharpe']:+.3f}"
        )

    val_path = os.path.join(SWEEP_RESULTS_DIR, "seed_sweep_validation.csv")
    val_df = _aggregate(val_rows, val_path)
    winner = val_df.sort_values("score", ascending=False).iloc[0]
    winner_seed = int(winner["seed"])

    print("\n" + "=" * 72)
    print("  Validation Selection Complete")
    print("=" * 72)
    print(f"  Winner seed : {winner_seed}")
    print(f"  Val score   : {winner['score']:+.4f}")
    print(f"  Val excess  : {winner['excess_return']*100:+.2f}% vs Equal-Weight")
    print(f"  CSV         : {val_path}")

    test_rows: list[dict[str, float]] = []
    if args.final_test:
        artifacts = artifacts_by_seed[winner_seed]
        print(f"\n  Final locked test evaluation for seed {winner_seed}")
        test_rows.append(_evaluate_seed(artifacts, "test", SWEEP_RESULTS_DIR))

    if args.eval_test_all:
        print("\n  Diagnostic test-all mode enabled; do not use these rows for selection.")
        for seed, artifacts in artifacts_by_seed.items():
            test_rows.append(_evaluate_seed(artifacts, "test", SWEEP_RESULTS_DIR))

    if test_rows:
        test_path = os.path.join(SWEEP_RESULTS_DIR, "seed_sweep_test.csv")
        _aggregate(test_rows, test_path)
        print(f"  Test CSV    : {test_path}")


if __name__ == "__main__":
    main()
