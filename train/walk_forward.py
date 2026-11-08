"""Walk-forward PPO training with validation checkpoint selection (protocol Section 4).

Every `eval_every` steps the deterministic policy runs on each validation year from
cash; the checkpoint with the best mean of the two validation Sharpe ratios is kept.
Test years never enter training or selection.

Run:
  python -m train.walk_forward --dev --folds 2019 --seeds 0 --steps 20000   # smoke run, 2014-2018 only
  python -m train.walk_forward --protocol                                   # all folds x seeds (gated)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from config import ROOT, describe, load, study_root
from train.common import RUNS, PPOPolicy, folds, protocol_gate, run_tag


def train_one(task: tuple) -> dict:
    fold_year, seed, steps, out_dir = task
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from data.panel import load_engine
    from env.engine import run
    from env.gym_env import RiskBudgetEnv
    from eval.metrics import sharpe

    cfg = load()
    data = load_engine()
    fold = folds(data, [fold_year])[0]
    pf, tr = cfg["portfolio"], cfg["training"]
    env = RiskBudgetEnv(data, fold.train_lo, fold.train_hi, fold.mu, fold.sd,
                        exposures=pf["exposure_grid"], horizons=pf["horizon_grid"],
                        kappa=cfg["reward"]["kappa"], scale=cfg["reward"]["scale"],
                        episode_sessions=tr["episode_sessions"], cost_bps=pf["cost_bps"],
                        clip=cfg["state"]["clip"])
    env.reset(seed=seed)
    out = study_root() / out_dir
    out.mkdir(parents=True, exist_ok=True)

    class ValCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best, self.best_step, self.rows = -np.inf, 0, []

        def _evaluate(self):
            pol = PPOPolicy(self.model, data, fold)
            shs = []
            for (_, s, e) in fold.val:
                res = run(data, pol, s, e, cost_bps=pf["cost_bps"])
                shs.append(sharpe(res.returns, res.rf))
            m = float(np.mean(shs))
            row = {"step": self.num_timesteps, "mean": m}
            row.update({f"val_{y}": v for (y, _, _), v in zip(fold.val, shs)})
            self.rows.append(row)
            if m > self.best:
                self.best, self.best_step = m, self.num_timesteps
                self.model.save(out / "best.zip")

        def _on_step(self):
            if self.num_timesteps % tr["eval_every"] == 0:
                self._evaluate()
            return True

    p = tr["ppo"]
    model = PPO("MlpPolicy", env, learning_rate=p["learning_rate"], n_steps=p["n_steps"],
                batch_size=p["batch_size"], n_epochs=p["n_epochs"], gamma=p["gamma"],
                gae_lambda=p["gae_lambda"], clip_range=p["clip_range"], ent_coef=p["ent_coef"],
                vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
                policy_kwargs={"net_arch": list(p["net_arch"])}, seed=seed, device="cpu", verbose=0)
    cb = ValCallback()
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=cb)
    secs = time.time() - t0
    pd.DataFrame(cb.rows).to_csv(out / "val_curve.csv", index=False)
    summary = {"fold": fold_year, "seed": seed, "steps": steps, "best_step": cb.best_step,
               "best_val_sharpe": cb.best, "seconds": round(secs, 1), "run_tag": run_tag()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true", help="development run on fold 2019 only")
    mode.add_argument("--protocol", action="store_true", help="the pre-registered run (gated)")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args(argv)
    cfg = load()
    tr = cfg["training"]

    if a.protocol:
        problems = protocol_gate()
        if problems:
            sys.exit("refusing protocol run:\n  - " + "\n  - ".join(problems))
        fold_years, seeds, steps = cfg["folds"]["test_years"], tr["seeds"], tr["total_steps"]
        base = "protocol"
    else:
        fold_years = a.folds or [2019]
        if any(y != 2019 for y in fold_years):
            sys.exit("dev runs are limited to fold 2019 (train 2014-2016, validate 2017-2018)")
        seeds, steps = a.seeds or [0], a.steps or tr["total_steps"]
        base = f"dev/{time.strftime('%Y%m%d-%H%M%S')}"

    tasks = [(y, s, steps, f"runs/{base}/fold_{y}/seed_{s}") for y in fold_years for s in seeds]
    workers = min(a.workers or tr["workers"], len(tasks))
    run_rel = (RUNS / base).relative_to(ROOT)
    print(f"protocol {describe()}: {len(tasks)} runs x {steps} steps on {workers} workers -> {run_rel}")
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(train_one, tasks):
            results.append(r)
            print(f"  fold {r['fold']} seed {r['seed']}: best val Sharpe {r['best_val_sharpe']:.3f} "
                  f"at {r['best_step']} steps ({r['seconds']:.0f}s)", flush=True)
    res = pd.DataFrame(results).sort_values(["fold", "seed"])
    res.to_csv(RUNS / base / "summary.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    from data.rf import is_provisional
    from train import ledger
    desc = f"PPO kappa={cfg['reward']['kappa']} steps={steps}"
    if cfg["version"] != 1:
        desc = f"protocol-v{cfg['version']} {cfg['data']['universe']}: {desc}"
    ledger.append("ppo_config" if a.protocol else "dev_smoke", desc,
                  folds=" ".join(map(str, fold_years)), seeds=" ".join(map(str, seeds)),
                  steps=str(steps), val_sharpe_mean=float(res["best_val_sharpe"].mean()),
                  touched_test=False, counts_as_trial=bool(a.protocol),
                  notes=(f"{run_rel}; rf={'PROVISIONAL' if is_provisional() else 'RBI'}"
                         + ("" if a.protocol else "; plumbing check with the protocol config, not a selectable variant")))


if __name__ == "__main__":
    main()
