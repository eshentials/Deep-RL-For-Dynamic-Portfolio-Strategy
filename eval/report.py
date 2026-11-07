"""Tables, statistics and figures from a test run (protocol Section 7).

Input: <out>/test_returns.parquet (+ extras.json, ensemble_decisions.csv, run_info.json).
Output: <out>/report.md, <out>/tables/*.csv, <out>/figures/*.png|pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import ROOT, load  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval.bootstrap import paired_metric_diff, paired_sharpe_diff  # noqa: E402
from eval.dsr import dsr  # noqa: E402

# Palette: first three validated categorical slots (light) + neutral for the benchmark.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {
    "ppo_ensemble": ("PPO ensemble", "#2a78d6", "-", 1.6),
    "vol_target_floored": ("Vol targeting", "#eb6834", "--", 1.3),
    "equal_weight": ("Equal weight", "#1baf7a", "-.", 1.3),
    "niftybees": ("NIFTYBEES", INK2, ":", 1.4),
}
LABELS = {"ppo_ensemble": "PPO ensemble", "vol_target_floored": "Vol targeting (floor 0.5)",
          "vol_target_unfloored": "Vol targeting (no floor)", "best_fixed": "Best fixed action",
          "equal_weight": "Equal weight (monthly)", "niftybees": "NIFTYBEES buy-and-hold",
          "rp_fixed_0.5": "Risk parity, e=0.50", "rp_fixed_0.7": "Risk parity, e=0.70",
          "rp_fixed_0.85": "Risk parity, e=0.85", "rp_fixed_1": "Risk parity, e=1.00"}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, width=0.6)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fig(w=7.0, h=3.0, **kw):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    for a in np.atleast_1d(ax):
        _style_axes(a)
    return fig, ax


def _direct_labels(ax, items, min_gap_px=10):
    """End-of-line labels, pushed apart vertically so they never overlap."""
    import matplotlib.dates as mdates
    items = [(lab, mdates.date2num(pd.Timestamp(x)) if not np.isscalar(x) or isinstance(x, pd.Timestamp) else x,
              float(y)) for lab, x, y in items]
    fig = ax.figure
    fig.canvas.draw()
    pts = sorted(((ax.transData.transform((x, y))[1], lab) for lab, x, y in items))
    placed = []
    for ypix, lab in pts:
        if placed and ypix - placed[-1][0] < min_gap_px:
            ypix = placed[-1][0] + min_gap_px
        placed.append((ypix, lab))
    x_end = max(x for _, x, _ in items)
    xpix = ax.transData.transform((x_end, 0))[0]
    for ypix, lab in placed:
        xd, yd = ax.transData.inverted().transform((xpix, ypix))
        ax.annotate(lab, (x_end, yd), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=INK, va="center", annotation_clip=False)


def _save(fig, out: Path, name: str):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def drawdown_episode(nav: pd.Series, year: int):
    """Deepest peak-to-trough episode of `nav` whose peak falls in `year`."""
    best = None
    peak_val, peak_date = -np.inf, None
    for d, v in nav.items():
        if v >= peak_val:
            peak_val, peak_date = v, d
        if peak_date is not None and peak_date.year == year:
            depth = v / peak_val - 1.0
            if best is None or depth < best[2]:
                best = (peak_date, d, depth)
    return best


def report(out: Path) -> dict:
    cfg = load()
    out = Path(out)
    rets = pd.read_parquet(out / "test_returns.parquet")
    rf = rets.pop("_rf").to_numpy()
    extras = json.loads((out / "extras.json").read_text())
    info = json.loads((out / "run_info.json").read_text())
    dec = pd.read_csv(out / "ensemble_decisions.csv", parse_dates=["date"])
    seeds = info["seeds"]
    dates = rets.index
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    named = [c for c in LABELS if c in rets] + [f"ppo_seed_{s}" for s in seeds]
    rand = [c for c in rets if c.startswith("random_")]
    sens = [c for c in rets if c.endswith(("_lag1", "_cost20", "_haircut"))]

    def row(c):
        e = extras.get(c, {})
        return M.summary(rets[c].to_numpy(), rf, costs=e.get("costs"), turnover=e.get("turnover"),
                         avg_exposure=e.get("avg_exposure"))
    combined = pd.DataFrame({c: row(c) for c in named + sens}).T
    rs = np.array([M.sharpe(rets[c].to_numpy(), rf) for c in rand])
    combined.loc["random (median of 100)", "sharpe"] = float(np.median(rs))
    combined.loc["random (median of 100)", "sessions"] = len(rets)
    combined.to_csv(tables / "combined.csv")

    b = cfg["evaluation"]["bootstrap"]
    h1 = paired_sharpe_diff(rets["ppo_ensemble"], rets["vol_target_floored"], rf,
                            block=b["block"], draws=b["draws"], seed=b["seed"])
    supported = h1["p05"] > 0

    expl = []
    for c in [c for c in named if c not in ("ppo_ensemble",)]:
        d = paired_sharpe_diff(rets["ppo_ensemble"], rets[c], rf, block=b["block"], draws=2000, seed=b["seed"])
        expl.append({"vs": c, "delta_sharpe": d["observed"], "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1]})
    expl = pd.DataFrame(expl)
    expl.to_csv(tables / "delta_sharpe_exploratory.csv", index=False)

    sec = []
    for c in ("equal_weight", "niftybees"):
        for name, fn in (("max_drawdown", M.max_drawdown), ("calmar", M.calmar)):
            d = paired_metric_diff(rets["ppo_ensemble"], rets[c], fn, block=b["block"], draws=2000, seed=b["seed"])
            sec.append({"vs": c, "metric": name, "delta": d["observed"], "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1]})
    sec = pd.DataFrame(sec)
    sec.to_csv(tables / "secondary.csv", index=False)

    from train import ledger
    tc = ledger.trial_counts()
    ex = rets["ppo_ensemble"].to_numpy() - rf
    d_this = dsr(ex, max(1, tc["this_study"]))
    d_prior = dsr(ex, max(1, tc["with_prior"]))

    prev = set(cfg["folds"]["previously_seen"])
    per_year = []
    for c in ["ppo_ensemble", "vol_target_floored", "vol_target_unfloored", "best_fixed", "equal_weight",
              "niftybees"] + [f"ppo_seed_{s}" for s in seeds]:
        if c not in rets:
            continue
        t = M.by_year(dates, rets[c].to_numpy(), rf)
        t["strategy"] = c
        per_year.append(t)
    per_year = pd.concat(per_year).rename_axis("year").reset_index()
    per_year["previously_seen"] = per_year["year"].isin(prev)
    per_year.to_csv(tables / "per_year.csv", index=False)

    yearly_d = []
    for y in sorted(set(dates.year)):
        m = dates.year == y
        d = paired_sharpe_diff(rets.loc[m, "ppo_ensemble"], rets.loc[m, "vol_target_floored"], rf[m],
                               block=b["block"], draws=2000, seed=b["seed"])
        yearly_d.append({"year": y, "delta": d["observed"], "lo": d["ci95"][0], "hi": d["ci95"][1]})
    yearly_d = pd.DataFrame(yearly_d)
    yearly_d.to_csv(tables / "per_year_delta_sharpe.csv", index=False)

    navs = {c: pd.Series(M.nav(rets[c].to_numpy())[1:], index=dates) for c in named}
    episodes = []
    for y in cfg["evaluation"]["drawdown_episode_years"]:
        ep = drawdown_episode(navs["niftybees"], y) if (dates.year == y).any() else None
        if ep is None:
            continue
        p, t, depth = ep
        for c in ["ppo_ensemble", "vol_target_floored", "vol_target_unfloored", "best_fixed", "equal_weight", "niftybees"]:
            if c in navs:
                seg = navs[c][(navs[c].index >= p) & (navs[c].index <= t)]
                episodes.append({"year": y, "peak": p.date(), "trough": t.date(), "strategy": c,
                                 "return_peak_to_trough": float(seg.iloc[-1] / seg.iloc[0] - 1),
                                 "max_dd_in_window": float((seg / seg.cummax() - 1).min())})
    episodes = pd.DataFrame(episodes)
    episodes.to_csv(tables / "drawdown_episodes.csv", index=False)

    _figures(out, rets, rf, dates, navs, dec, yearly_d, per_year, seeds)

    res = {"mode": info["mode"], "path": f"{info['start']} .. {info['end']}", "sessions": len(rets),
           "h1": {"delta_sharpe": h1["observed"], "p05": h1["p05"], "ci95": h1["ci95"],
                  "share_le_0": h1["share_le_0"], "supported": bool(supported)},
           "dsr_this_study": d_this, "dsr_with_prior": d_prior, "trial_counts": tc,
           "random_sharpe": {"median": float(np.median(rs)), "p05": float(np.quantile(rs, 0.05)),
                             "p95": float(np.quantile(rs, 0.95))},
           "delisting": extras.get("_delisting_haircut")}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    _markdown(out, res, combined, expl, sec, per_year, yearly_d, episodes)
    return res


def _figures(out, rets, rf, dates, navs, dec, yearly_d, per_year, seeds):
    vix = pd.read_parquet(ROOT / "data" / "processed" / "market.parquet")["vix"].reindex(dates)

    from matplotlib.ticker import FuncFormatter, NullFormatter
    fig, ax = _fig(7.0, 3.2)
    ends = []
    for c, (lab, col, ls, lw) in STYLE.items():
        if c in navs:
            ax.plot(navs[c].index, navs[c].values, color=col, ls=ls, lw=lw, label=lab)
            ends.append((lab, navs[c].index[-1], navs[c].values[-1]))
    from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator
    ax.set_yscale("log")
    lo = min(float(navs[c].min()) for c in STYLE if c in navs)
    hi = max(float(navs[c].max()) for c in STYLE if c in navs)
    ticks = [t for t in MaxNLocator(nbins=6).tick_values(lo, hi) if lo * 0.98 <= t <= hi * 1.02]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Growth of 1 (log scale)", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _direct_labels(ax, ends)
    _save(fig, out, "equity")

    fig, ax = _fig(7.0, 2.6)
    for c, (lab, col, ls, lw) in STYLE.items():
        if c in navs:
            dd = navs[c] / navs[c].cummax() - 1
            ax.plot(dd.index, dd.values * 100, color=col, ls=ls, lw=lw, label=lab)
    ax.set_ylabel("Drawdown (%)", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=4)
    _save(fig, out, "drawdown")

    # Two panels on one time axis (never a dual y-axis).
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1, 1.2]})
    for a in (a1, a2):
        _style_axes(a)
    a1.plot(vix.index, vix.values, color=INK2, lw=1.0)
    a1.set_ylabel("India VIX", fontsize=8, color=INK)
    if len(dec):
        s = pd.Series(dec["exposure"].values, index=dec["date"]).reindex(dates, method="ffill")
        a2.step(s.index, s.values, where="post", color="#2a78d6", lw=1.4)
    a2.set_ylabel("Ensemble exposure", fontsize=8, color=INK)
    a2.set_ylim(0.45, 1.02)
    _save(fig, out, "exposure_vs_vix")

    fig, ax = _fig(7.0, 2.6)
    yd = yearly_d
    ax.axhline(0, color=INK2, lw=0.8)
    ax.errorbar(yd["year"], yd["delta"], yerr=[yd["delta"] - yd["lo"], yd["hi"] - yd["delta"]],
                fmt="o", color="#2a78d6", ecolor="#2a78d6", elinewidth=1.2, capsize=3, ms=5)
    ax.set_xticks(yd["year"])
    ax.set_ylabel("Sharpe difference\n(PPO ensemble - vol targeting)", fontsize=8, color=INK)
    _save(fig, out, "per_year_delta_sharpe")

    fig, ax = _fig(7.0, 2.6)
    py = per_year
    seed_rows = py[py["strategy"].str.startswith("ppo_seed_")]
    ens = py[py["strategy"] == "ppo_ensemble"]
    ax.scatter(seed_rows["year"], seed_rows["sharpe"], s=18, color="#9b9a96", label="Single seeds", zorder=2)
    ax.scatter(ens["year"], ens["sharpe"], s=34, marker="D", color="#2a78d6", label="Ensemble", zorder=3,
               edgecolors=SURFACE, linewidths=1.0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(sorted(py["year"].unique()))
    ax.set_ylabel("Sharpe ratio", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="best")
    _save(fig, out, "seed_spread")


def _fmt(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x * 100:.2f}%" if pct else f"{x:.{nd}f}"


def _markdown(out, res, combined, expl, sec, per_year, yearly_d, episodes):
    h = res["h1"]
    lines = [f"# Test report ({res['mode']})", "",
             f"Path: {res['path']} ({res['sessions']} sessions).", "",
             "## Primary hypothesis (H1)", "",
             f"Sharpe(PPO ensemble) - Sharpe(vol targeting, floor 0.5) = **{h['delta_sharpe']:+.3f}**; "
             f"bootstrap 5th percentile {h['p05']:+.3f}; 95% CI [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]; "
             f"share of draws <= 0: {h['share_le_0']:.3f}.", "",
             f"**H1 {'SUPPORTED' if h['supported'] else 'NOT SUPPORTED'}** under the pre-registered rule "
             "(5th percentile above zero).", "",
             "## Combined path", "",
             "| Strategy | Sharpe | CAGR | Vol | Max DD | Calmar | Costs | Avg exposure |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in combined.iterrows():
        lines.append(f"| {LABELS.get(name, name)} | {_fmt(r['sharpe'])} | {_fmt(r.get('cagr'), True)} | "
                     f"{_fmt(r.get('vol'), True)} | {_fmt(r.get('max_drawdown'), True)} | {_fmt(r.get('calmar'))} | "
                     f"{_fmt(r.get('costs'), True)} | {_fmt(r.get('avg_exposure'), nd=2)} |")
    lines += ["", "## Deflated Sharpe Ratio (PPO ensemble)", "",
              f"- This study's trials (N={res['dsr_this_study']['n_trials']}): DSR {res['dsr_this_study']['dsr']:.3f}, "
              f"PSR {res['dsr_this_study']['psr']:.3f}",
              f"- With recorded prior work (N={res['dsr_with_prior']['n_trials']}): DSR {res['dsr_with_prior']['dsr']:.3f}",
              "", "## Secondary outcomes (exploratory intervals)", "",
              "| vs | Metric | PPO - baseline | 95% CI |", "|---|---|---|---|"]
    for r in sec.itertuples():
        lines.append(f"| {LABELS.get(r.vs, r.vs)} | {r.metric} | {r.delta:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |")
    lines += ["", "## Sharpe differences vs other baselines (exploratory)", "",
              "| vs | PPO - baseline | 95% CI |", "|---|---|---|"]
    for r in expl.itertuples():
        lines.append(f"| {LABELS.get(r.vs, r.vs)} | {r.delta_sharpe:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |")
    rnd = res["random_sharpe"]
    lines += ["", f"Random policy Sharpe: median {rnd['median']:.3f}, 5-95% band [{rnd['p05']:.3f}, {rnd['p95']:.3f}].",
              "", "## Per year (previously seen years marked *)", "",
              "| Year | PPO ens. | Vol tgt | Best fixed | Equal wt | NIFTYBEES | PPO - VT [95% CI] |",
              "|---|---|---|---|---|---|---|"]
    pv = per_year.pivot_table(index="year", columns="strategy", values="sharpe")
    seen = per_year.drop_duplicates("year").set_index("year")["previously_seen"]
    for y in pv.index:
        d = yearly_d.set_index("year").loc[y]
        lines.append(f"| {y}{'*' if seen.get(y) else ''} | {_fmt(pv.loc[y].get('ppo_ensemble'))} | "
                     f"{_fmt(pv.loc[y].get('vol_target_floored'))} | {_fmt(pv.loc[y].get('best_fixed'))} | "
                     f"{_fmt(pv.loc[y].get('equal_weight'))} | {_fmt(pv.loc[y].get('niftybees'))} | "
                     f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] |")
    if len(episodes):
        lines += ["", "## Drawdown episodes (NIFTYBEES's deepest episode with its peak in the year)", "",
                  "| Year | Peak | Trough | Strategy | Peak-to-trough | Max DD in window |", "|---|---|---|---|---|---|"]
        for r in episodes.itertuples():
            lines.append(f"| {r.year} | {r.peak} | {r.trough} | {LABELS.get(r.strategy, r.strategy)} | "
                         f"{r.return_peak_to_trough * 100:+.2f}% | {r.max_dd_in_window * 100:.2f}% |")
    dl = res.get("delisting") or {}
    lines += ["", f"Delisting haircut sensitivity: {'applied' if dl.get('applicable') else 'not applicable'} "
              f"({dl.get('delisting_exits_on_path', 0)} delisting exits on the path).", "",
              "Figures: figures/equity, drawdown, exposure_vs_vix, per_year_delta_sharpe, seed_spread (png and pdf)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
