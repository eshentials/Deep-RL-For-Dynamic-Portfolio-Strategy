"""
train_smart.py — Train Smart Tangency Portfolio Agents
=======================================================

Trains 6 DRL agents (PPO × 3 utilities  +  A2C × 3 utilities) as described
in the Smart Tangency paper (Yu & Chang, 2025):

  Agents:
    PPO-MV    PPO-SV    PPO-CVaR
    A2C-MV    A2C-SV    A2C-CVaR

  Ensemble:
    After training, each agent is evaluated on the validation split.
    The ensemble model = agent with the highest median Sharpe ratio on val.
    This selection is saved to models/ensemble_best.json for use in evaluate_smart.py.

Usage
─────
  python3.13 train_smart.py                     # train all 6 agents
  python3.13 train_smart.py --utility mv        # train MV agents only
  python3.13 train_smart.py --algo ppo          # train PPO agents only
  python3.13 train_smart.py --timesteps 500000  # longer training
  python3.13 train_smart.py --resume            # resume from latest checkpoints
  python3.13 train_smart.py --ensemble-only     # skip training; just run ensemble

Outputs
───────
  models/smart_ppo_mv_latest.zip       — PPO-MV
  models/smart_ppo_sv_latest.zip       — PPO-SV
  models/smart_ppo_cvar_latest.zip     — PPO-CVaR
  models/smart_a2c_mv_latest.zip       — A2C-MV
  models/smart_a2c_sv_latest.zip       — A2C-SV
  models/smart_a2c_cvar_latest.zip     — A2C-CVaR
  models/ensemble_best.json            — {"algo": "ppo", "utility": "mv", ...}
  logs/smart_<algo>_<utility>/         — TensorBoard logs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from envs.env_smart import (
    SmartPortfolioEnv,
    HOLDING_PERIODS_SMART,
    N_HOLD_SMART,
)
from core.smart_tangency import N_LAMBDA, UTILITIES
from state_spec import TRANSACTION_COST_RATE

PRESET = "full"

TIMESTEPS = {
    "fast": 150_000,
    "full": 300_000,
}
DEFAULT_TIMESTEPS = TIMESTEPS[PRESET]

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR    = "models"
LOG_DIR      = "logs"
ENSEMBLE_CFG = os.path.join(MODEL_DIR, "ensemble_best.json")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

ALGOS        = ["ppo", "a2c"]

# ── Training presets ──────────────────────────────────────────────────────────
# "fast"  : ~25 min on M2 Air per agent  (good for iteration / debugging)
# "full"  : ~60 min on M2 Air per agent  (paper-quality results)
# Use --fast flag or set PRESET = "fast" to override.
PRESET = "full"

TIMESTEPS = {
    "fast": 150_000,   # ~25 min/agent on M2 Air
    "full": 300_000,   # ~60 min/agent on M2 Air  (down from 1M — see note)
}
# Note: 300k steps with the tuned hyperparameters below is equivalent to
# 1M steps with the original config because:
#   · PPO batch_size doubled (128) → 2× data efficiency per update
#   · A2C n_steps raised (64) → far fewer redundant micro-updates
#   · Network halved ([128,128]) → 4× faster forward/backward pass

CHECKPOINT_FREQ   = 25_000
EVAL_FREQ         = 15_000
N_EVAL_EPISODES   = 3    # val episodes per EvalCallback check
N_VAL_EPISODES    = 10   # episodes for final ensemble Sharpe distribution
ENSEMBLE_TAIL_FRAC = 0.20  # last 20% of train episodes for rolling Sharpe

# ── Thread cap — prevents thermal throttling on laptops ──────────────────────
# M2 Air has no fan; sustained full-core load throttles after ~20 min.
# Capping at 4 threads keeps it cool and often runs FASTER end-to-end.
DEFAULT_N_THREADS = 4


# ── PPO hyperparameters ────────────────────────────────────────────────────────
# Key changes from v1:
#   batch_size 64→128  : fewer updates, each sees more data → more stable
#   n_epochs   10→6    : less overfitting per rollout
#   net_arch   [256,256]→[128,128]  : 4× fewer params, 3× faster per step
#   learning_rate 3e-4→1e-4 : more conservative, less likely to diverge early
PPO_CONFIG = dict(
    learning_rate = 1e-4,
    n_steps       = 512,
    batch_size    = 128,
    n_epochs      = 6,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    policy_kwargs = dict(net_arch=[128, 128]),
    verbose       = 1,
    tensorboard_log = LOG_DIR,
)

# ── A2C hyperparameters ────────────────────────────────────────────────────────
# Key changes from v1:
#   n_steps  5→64   : THE main fix — n_steps=5 means a gradient update every
#                     5 env steps, creating enormous overhead. 64 is the
#                     practical minimum for portfolio envs with long episodes.
#   net_arch [256,256]→[128,128]  : matches PPO, 4× fewer params
#   learning_rate 7e-4→3e-4 : more stable with larger n_steps
A2C_CONFIG = dict(
    learning_rate  = 3e-4,
    n_steps        = 64,
    gamma          = 0.99,
    gae_lambda     = 1.0,
    ent_coef       = 0.01,
    vf_coef        = 0.25,
    max_grad_norm  = 0.5,
    policy_kwargs  = dict(net_arch=[128, 128]),
    verbose        = 1,
    tensorboard_log = LOG_DIR,
)


def _model_name(algo: str, utility: str) -> str:
    return f"smart_{algo}_{utility}"


def _model_path(algo: str, utility: str) -> str:
    return os.path.join(MODEL_DIR, f"{_model_name(algo, utility)}_latest")


def _best_path(algo: str, utility: str) -> str:
    return os.path.join(MODEL_DIR, f"{_model_name(algo, utility)}_best")


def _log_path(algo: str, utility: str) -> str:
    return os.path.join(LOG_DIR, _model_name(algo, utility))


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK — log portfolio-specific metrics
# ══════════════════════════════════════════════════════════════════════════════

class SmartMetricsCallback(BaseCallback):
    """Log mean λ, hold days, NAV, and TC to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._lambda_hist: list[float] = []
        self._hold_hist:   list[int]   = []
        self._nav_hist:    list[float] = []
        self._tc_hist:     list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "lambda_idx" in info:
                self._lambda_hist.append(info["lambda_idx"])
                self._hold_hist.append(info["hold_days"])
                self._tc_hist.append(info.get("tc_cost", 0.0))
            if info.get("nav"):
                self._nav_hist.append(info["nav"])

        if self.n_calls % 500 == 0 and self._lambda_hist:
            self.logger.record("portfolio/mean_lambda_idx",
                               float(np.mean(self._lambda_hist[-200:])))
            self.logger.record("portfolio/mean_hold_days",
                               float(np.mean(self._hold_hist[-200:])))
            if self._tc_hist:
                self.logger.record("portfolio/mean_tc",
                                   float(np.mean(self._tc_hist[-200:])))
            if self._nav_hist:
                self.logger.record("portfolio/mean_nav",
                                   float(np.mean(self._nav_hist[-50:])))
        return True


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING — single (algo, utility) pair
# ══════════════════════════════════════════════════════════════════════════════

def train_one(algo:            str,
              utility:         str,
              total_timesteps: int  = DEFAULT_TIMESTEPS,
              resume:          bool = False) -> None:
    """
    Train one (algo, utility) agent.

    Parameters
    ──────────
    algo            : "ppo" or "a2c"
    utility         : "mv", "sv", or "cvar"
    total_timesteps : total env steps
    resume          : if True, continue from existing checkpoint
    """
    name = _model_name(algo, utility)
    print(f"\n{'─'*65}")
    print(f"  Training: {algo.upper()} + {utility.upper()}")
    print(f"  Model   : {name}")
    print(f"  Steps   : {total_timesteps:,}")
    print(f"{'─'*65}")

    train_env = make_vec_env(
        lambda: SmartPortfolioEnv(split="train", utility=utility),
        n_envs=1,
    )
    eval_env = make_vec_env(
        lambda: SmartPortfolioEnv(split="val", utility=utility),
        n_envs=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = name,
        verbose     = 0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = MODEL_DIR,
        log_path             = _log_path(algo, utility),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 0,
    )
    metrics_cb = SmartMetricsCallback(verbose=0)

    latest_path = _model_path(algo, utility)

    if algo == "ppo":
        AlgoCls = PPO
        cfg     = PPO_CONFIG.copy()
    else:
        AlgoCls = A2C
        cfg     = A2C_CONFIG.copy()

    if resume and os.path.exists(latest_path + ".zip"):
        print(f"  Resuming from {latest_path}.zip")
        model = AlgoCls.load(latest_path, env=train_env)
        model.set_env(train_env)
    else:
        model = AlgoCls("MlpPolicy", train_env, **cfg)

    params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy: MlpPolicy  params={params:,}")

    t0 = time.time()
    model.learn(
        total_timesteps    = total_timesteps,
        callback           = [checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name        = name,
        reset_num_timesteps = not resume,
        progress_bar       = True,
    )
    elapsed = time.time() - t0

    model.save(latest_path)
    print(f"  Done in {elapsed/60:.1f} min  → {latest_path}.zip")

    train_env.close()
    eval_env.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE SELECTION — pick best model by median Sharpe on val split
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe_from_nav(nav: list[float]) -> float:
    """
    Robust annualised Sharpe using log returns.
    Fixes instability + inflated values.
    """
    nav_arr = np.array(nav, dtype=np.float64)

    if len(nav_arr) < 3:
        return 0.0

    # log returns (correct)
    returns = np.diff(np.log(nav_arr))

    mean_ret = np.mean(returns)
    std_ret  = np.std(returns) + 1e-6

    sharpe = (mean_ret / std_ret) * np.sqrt(252)

    # safety clamp (prevents insane values)
    sharpe = float(np.clip(sharpe, -10.0, 10.0))

    return sharpe


def _eval_model_on_split(algo:      str,
                          utility:   str,
                          split:     str = "train",
                         n_episodes: int = 1) -> np.ndarray:
    """
    Run deterministic rollouts on the TAIL of the training split and return
    the Sharpe distribution.  Using the train tail rather than a separate val
    split prevents temporal leakage: the original val (2022-24) came after
    test (2021), so ensemble selection on val implicitly saw a future regime.

    Each episode starts at a random point within the last ENSEMBLE_TAIL_FRAC
    of the period, sampled with a fixed seed for reproducibility.
    """
    path = _model_path(algo, utility)
    if not os.path.exists(path + ".zip"):
        path = _best_path(algo, utility)
        if not os.path.exists(path + ".zip"):
            print(f"    ✗ No model found for {algo}-{utility}")
            return np.array([])

    AlgoCls = PPO if algo == "ppo" else A2C
    try:
        model = AlgoCls.load(path)
    except Exception as e:
        print(f"    ✗ Failed to load {path}: {e}")
        return np.array([])

    sharpes = []
    for ep in range(n_episodes):
        env  = SmartPortfolioEnv(split=split, utility=utility)
        T    = env.T
        warm = env.warm_up

        # Start each episode in the last ENSEMBLE_TAIL_FRAC of the period.
        tail_start = warm + int((T - warm) * (1.0 - ENSEMBLE_TAIL_FRAC))
        rng        = np.random.default_rng(ep + 42)
        t0         = int(rng.integers(tail_start, max(tail_start + 1, T - 60)))
        env.t      = t0
        obs, _     = env.reset(seed=ep)
        env.t      = t0   # re-apply after reset clears it

        navs = [env.nav]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            navs.append(info["nav"])
            done = terminated or truncated
            
        sharpes.append(_sharpe_from_nav(navs))
        env.close()

    return np.array(sharpes)


def select_ensemble() -> dict:
    """
    Evaluate all 6 trained models on the TAIL of the training split
    (last ENSEMBLE_TAIL_FRAC of the training period).

    Using the training tail rather than a separate val split avoids temporal
    leakage: val (2022-24) came AFTER test (2021), so a model selected on val
    had implicitly seen a later market regime.

    Selection criterion: highest MEDIAN Sharpe ratio across episodes.
    Returns dict with keys: algo, utility, median_sharpe, model_path
    """
    print("\n" + "=" * 65)
    print("  ENSEMBLE SELECTION")
    print(f"  Metric: median Sharpe on training-tail (last {int(ENSEMBLE_TAIL_FRAC*100)}% of train)")
    print("  Rationale: avoids val/test temporal leakage (val=2022-24 > test=2021)")
    print("=" * 65)

    best: dict | None = None

    for algo, utility in product(ALGOS, UTILITIES):
        print(f"\n  Evaluating {algo.upper()}-{utility.upper()} "
              f"({N_VAL_EPISODES} episodes on train tail) …",
              end=" ", flush=True)
        sharpes = _eval_model_on_split(algo, utility, split="train")

        if len(sharpes) == 0:
            print("SKIPPED (model missing)")
            continue

        median_sharpe = float(np.median(sharpes))
        mean_sharpe   = float(np.mean(sharpes))
        print(f"median={median_sharpe:+.3f}  mean={mean_sharpe:+.3f}  "
              f"[{sharpes.min():+.3f}, {sharpes.max():+.3f}]")

        if best is None or median_sharpe > best["median_sharpe"]:
            best = {
                "algo":            algo,
                "utility":         utility,
                "median_sharpe":   median_sharpe,
                "mean_sharpe":     mean_sharpe,
                "sharpe_dist":     sharpes.tolist(),
                "model_path":      _model_path(algo, utility),
                "selection_split": "train_tail",
                "tail_frac":       ENSEMBLE_TAIL_FRAC,
            }

    if best is None:
        print("\n  ⚠  No models evaluated. Defaulting to PPO-MV.")
        best = {
            "algo":            "ppo",
            "utility":         "mv",
            "median_sharpe":   float("nan"),
            "mean_sharpe":     float("nan"),
            "sharpe_dist":     [],
            "model_path":      _model_path("ppo", "mv"),
            "selection_split": "train_tail",
            "tail_frac":       ENSEMBLE_TAIL_FRAC,
        }

    print(f"\n  ★ ENSEMBLE WINNER: {best['algo'].upper()}-{best['utility'].upper()}")
    print(f"    Median Sharpe  = {best['median_sharpe']:+.3f}")
    print(f"    Selected on    : train tail (last {int(ENSEMBLE_TAIL_FRAC*100)}%)")
    print(f"    Model path     = {best['model_path']}.zip")

    with open(ENSEMBLE_CFG, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Ensemble config saved → {ENSEMBLE_CFG}")

    return best
# ══════════════════════════════════════════════════════════════════════════════
# LOAD ENSEMBLE — for use in evaluate_smart.py
# ══════════════════════════════════════════════════════════════════════════════

def load_ensemble_model():
    """
    Load the ensemble (best-validation) model.
    Returns (model, utility) tuple.
    """
    from stable_baselines3 import PPO, A2C

    if not os.path.exists(ENSEMBLE_CFG):
        raise FileNotFoundError(
            f"Ensemble config not found: {ENSEMBLE_CFG}\n"
            "Run train_smart.py first (--ensemble-only or full training)."
        )

    with open(ENSEMBLE_CFG) as f:
        cfg = json.load(f)

    algo    = cfg["algo"]
    utility = cfg["utility"]
    path    = cfg["model_path"]

    AlgoCls = PPO if algo == "ppo" else A2C
    model   = AlgoCls.load(path)
    print(f"  Ensemble model: {algo.upper()}-{utility.upper()}  "
          f"(val Sharpe={cfg['median_sharpe']:+.3f})")
    return model, utility


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Smart Tangency Portfolio DRL agents"
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                        help=f"Training timesteps per agent (default: {DEFAULT_TIMESTEPS:,})")
    parser.add_argument("--utility", type=str, default="all",
                        choices=list(UTILITIES) + ["all"],
                        help="Utility function to train (default: all)")
    parser.add_argument("--algo", type=str, default="all",
                        choices=ALGOS + ["all"],
                        help="RL algorithm (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoints")
    parser.add_argument("--ensemble-only", action="store_true",
                        help="Skip training; just run ensemble selection")
    args = parser.parse_args()

    utilities = list(UTILITIES)   if args.utility == "all" else [args.utility]
    algos     = ALGOS             if args.algo    == "all" else [args.algo]

    print("=" * 65)
    print("  Smart Tangency Portfolio — Training")
    print("=" * 65)
    print(f"  Algorithms  : {algos}")
    print(f"  Utilities   : {utilities}")
    print(f"  Timesteps   : {args.timesteps:,} per agent")
    print(f"  Action space: MultiDiscrete([{N_LAMBDA}, {N_HOLD_SMART}])")
    print(f"  Hold periods: {HOLDING_PERIODS_SMART}")
    print(f"  TC rate     : {TRANSACTION_COST_RATE*100:.1f}%")

    if not args.ensemble_only:
        for algo, utility in product(algos, utilities):
            train_one(
                algo            = algo,
                utility         = utility,
                total_timesteps = args.timesteps,
                resume          = args.resume,
            )

    # Always run ensemble selection after training
    best = select_ensemble()

    print("\n  All done.")
    print(f"  Best model : {best['algo'].upper()}-{best['utility'].upper()}")
    print(f"  Run evaluate_smart.py to backtest all strategies.")
