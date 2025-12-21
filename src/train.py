# train.py
import os
import csv
import math
import random
from collections import deque, defaultdict
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from agent import QAgent
from environment import MultiAgentMatrixGameEnv, ResourceSharingEnv
from config import (
    LR, GAMMA, EPS_START, EPS_END, EPS_DECAY,
    STEPS, EVAL_WINDOW, SEED,
    EMPATHY_ALPHAS, COOP_THRESHOLD,
    LOG_ROOT, EXPERIMENTS,
    SEEDS, BASELINES,ALPHA_BLOCK,
)

# ---- Optional new config knobs (safe defaults if missing) ----
try:
    from config import HETERO_PROFILES
except Exception:
    HETERO_PROFILES = []  # if empty we'll sample randomly

try:
    from config import ALPHA_CANDIDATES
except Exception:
    ALPHA_CANDIDATES = [0.0, 0.2, 0.5, 0.8, 1.0]

try:
    from config import ALPHA_BANDIT_EPS
except Exception:
    ALPHA_BANDIT_EPS = 0.1

try:
    from config import LEARN_ALPHA_SIGNAL
except Exception:
    LEARN_ALPHA_SIGNAL = "team"  # "self" | "team" | "shaped"


# -----------------------------
# small utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(window: deque) -> float:
    return float(sum(window)) / float(len(window))


def stdev(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sem(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return stdev(xs) / math.sqrt(len(xs))


# -----------------------------
# "cooperation" flags
# -----------------------------
def coop_flag_matrix(joint_actions: List[Any], coop_action: Any) -> int:
    return 1 if all(a == coop_action for a in joint_actions) else 0


def coop_flag_resource(extra: Dict[str, Any], sustainable_threshold: int) -> int:
    # coop if total extraction <= threshold
    return 1 if float(extra["total_extraction"]) <= float(sustainable_threshold) else 0


# -----------------------------
# CSV reading + aggregation
# -----------------------------
def read_series(csv_path: str, field: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get(field, "")
            if v == "" or v is None:
                continue
            xs.append(int(row["step"]))
            ys.append(float(v))
    return xs, ys


def aggregate_series(paths: List[str], field: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Returns (steps, mean, ci95) aligned by step index.
    Uses only steps present in all runs for that field.
    """
    per_run = []
    for p in paths:
        xs, ys = read_series(p, field)
        per_run.append((xs, ys))

    if not per_run:
        return [], [], []

    step_sets = [set(xs) for xs, _ in per_run]
    common_steps = sorted(set.intersection(*step_sets)) if step_sets else []
    if not common_steps:
        return [], [], []

    run_maps = [{x: y for x, y in zip(xs, ys)} for xs, ys in per_run]

    means, ci95 = [], []
    for s in common_steps:
        vals = [rm[s] for rm in run_maps]
        m = sum(vals) / len(vals)
        ci = 1.96 * sem(vals)
        means.append(m)
        ci95.append(ci)

    return common_steps, means, ci95


import seaborn as sns

def setup_plot_style(style: str = "whitegrid", context: str = "paper", font_scale: float = 1.1):
    """Configure matplotlib and seaborn for publication-grade plots."""
    sns.set_theme(style=style, context=context, font_scale=font_scale)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "black",
        "legend.borderpad": 0.8,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
    })


def plot_metric_with_ci(
    cond_paths: Dict[str, List[str]],
    out_dir: str,
    title: str,
    field: str,
    ylabel: str,
    out_name: str,
    xlabel: str = "Training Steps",
    figsize: Tuple[float, float] = (10, 6),
    show_markers: bool = False,
    marker_every: int = 10,
    log_scale: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Create publication-grade plot with confidence intervals.
    
    Args:
        cond_paths: Dict mapping condition names to lists of CSV file paths
        out_dir: Output directory for saving figure
        title: Plot title
        field: Column name to plot
        ylabel: Y-axis label
        out_name: Output filename
        xlabel: X-axis label (default: "Training Steps")
        figsize: Figure size as (width, height)
        show_markers: Whether to show markers on lines
        marker_every: Show markers every N steps (if show_markers=True)
        log_scale: Use log scale for y-axis
        ylim: Y-axis limits as (min, max)
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("husl", n_colors=len(cond_paths))
    
    for i, (cond_name, paths) in enumerate(cond_paths.items()):
        all_series = []
        for p in paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    if field in df.columns:
                        all_series.append(df[field].values)
                except Exception as e:
                    print(f"Warning: Could not read {p}: {e}")
        
        if not all_series:
            print(f"Warning: No valid data found for condition '{cond_name}'")
            continue
        
        # Align all series to minimum length
        min_len = min(len(s) for s in all_series)
        data = np.array([s[:min_len] for s in all_series])
        steps = np.arange(min_len)
        
        # Compute statistics
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        ci = 1.96 * std / np.sqrt(len(all_series))  # 95% CI
        
        # Plot line
        line_kwargs = {"label": cond_name, "color": palette[i], "linewidth": 2.5}
        if show_markers:
            line_kwargs.update({"marker": "o", "markevery": marker_every, "markersize": 6})
        
        ax.plot(steps, mean, **line_kwargs)
        
        # Plot confidence interval as shaded region
        ax.fill_between(steps, mean - ci, mean + ci, color=palette[i], alpha=0.15)
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    
    if log_scale:
        ax.set_yscale("log")
    
    if ylim:
        ax.set_ylim(ylim)
    
    # Legend
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fontsize=10,
        framealpha=0.95,
        edgecolor="black"
    )
    
    # Spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, out_name), bbox_inches="tight", facecolor="white")
    plt.close()


def plot_multiple_metrics_grid(
    cond_paths: Dict[str, List[str]],
    out_dir: str,
    metrics: List[Tuple[str, str, str]],  # (field, ylabel, display_name)
    main_title: str,
    out_name: str,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Create a grid of plots for multiple metrics.
    
    Args:
        cond_paths: Dict mapping condition names to lists of CSV file paths
        out_dir: Output directory
        metrics: List of (field, ylabel, display_name) tuples
        main_title: Overall title for the figure
        out_name: Output filename
        ncols: Number of columns in grid
        figsize: Figure size (auto-calculated if None)
    """
    setup_plot_style()
    
    nrows = (len(metrics) + ncols - 1) // ncols
    if figsize is None:
        figsize = (6 * ncols, 4.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1) if ncols > 1 else np.array([[axes]])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    palette = sns.color_palette("husl", n_colors=len(cond_paths))
    
    for idx, (field, ylabel, display_name) in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]
        
        for i, (cond_name, paths) in enumerate(cond_paths.items()):
            all_series = []
            for p in paths:
                if os.path.exists(p):
                    try:
                        df = pd.read_csv(p)
                        if field in df.columns:
                            all_series.append(df[field].values)
                    except Exception as e:
                        print(f"Warning: Could not read {p}: {e}")
            
            if not all_series:
                continue
            
            min_len = min(len(s) for s in all_series)
            data = np.array([s[:min_len] for s in all_series])
            steps = np.arange(min_len)
            
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ci = 1.96 * std / np.sqrt(len(all_series))
            
            ax.plot(steps, mean, label=cond_name, color=palette[i], linewidth=2.5)
            ax.fill_between(steps, mean - ci, mean + ci, color=palette[i], alpha=0.15)
        
        ax.set_title(display_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    
    # Hide empty subplots
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    
    fig.suptitle(main_title, fontsize=16, fontweight="bold", y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(cond_paths), frameon=True, fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, out_name), bbox_inches="tight", facecolor="white")
    plt.close()


def plot_metric_comparison_bars(
    data: Dict[str, List[float]],
    out_dir: str,
    title: str,
    ylabel: str,
    out_name: str,
    xlabel: str = "Condition",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Create a bar plot with error bars for comparing conditions.
    
    Args:
        data: Dict mapping condition names to lists of values
        out_dir: Output directory
        title: Plot title
        ylabel: Y-axis label
        out_name: Output filename
        xlabel: X-axis label
        figsize: Figure size
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("husl", n_colors=len(data))
    
    conditions = list(data.keys())
    means = [np.mean(v) for v in data.values()]
    stds = [np.std(v) for v in data.values()]
    
    bars = ax.bar(conditions, means, yerr=stds, capsize=10, alpha=0.8, color=palette,
                   edgecolor="black", linewidth=1.5, error_kw={"linewidth": 2})
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--", linewidth=0.8)
    
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, out_name), bbox_inches="tight", facecolor="white")
    plt.close()


def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# hetero profile helper
# -----------------------------
def get_hetero_profile(N: int, seed: int) -> Tuple[List[float], str]:
    """
    Returns (profile_alphas, profile_tag).
    If HETERO_PROFILES exist and one matches length N, sample one.
    Otherwise sample per-agent alpha from ALPHA_CANDIDATES.
    """
    rng = random.Random(seed + 99991)

    candidates = [p for p in HETERO_PROFILES if isinstance(p, list) and len(p) == N]
    if candidates:
        prof = rng.choice(candidates)
        tag = "profile_" + "_".join(f"{a:.2f}" for a in prof)
        return [float(a) for a in prof], tag

    prof = [float(rng.choice(ALPHA_CANDIDATES)) for _ in range(N)]
    tag = "sampled_" + "_".join(f"{a:.2f}" for a in prof)
    return prof, tag


def learn_alpha_signal(
    mode: str,
    rewards: List[float],
    total_reward: float,
    N: int,
    agent_alpha: float,
    i: int,
) -> float:
    """
    Signal used to update alpha-bandit in learn_alpha baseline.
    mode: "self" | "team" | "shaped"
    """
    if mode == "self":
        return float(rewards[i])

    if mode == "team":
        return float(total_reward) / float(N)

    if mode == "shaped":
        r_self = float(rewards[i])
        r_others_avg = (float(total_reward) - r_self) / float(N - 1) if N > 1 else 0.0
        a = float(agent_alpha)
        return a * r_self + (1.0 - a) * r_others_avg

    raise ValueError(f"Unknown LEARN_ALPHA_SIGNAL: {mode}")


# -----------------------------
# one run (exp, baseline, alpha, seed)
# -----------------------------
def run_one_experiment(
    exp: Dict[str, Any],
    empathy_alpha: float,
    seed: int,
    baseline: str,
) -> Tuple[str, Optional[int]]:
    name = exp["name"]

    # log folder tags
    alpha_tag = f"alpha_{empathy_alpha:.2f}" if baseline == "empathy" else "alpha_na"
    run_dir = os.path.join(LOG_ROOT, name, baseline, alpha_tag, f"seed_{seed}")
    ensure_dir(run_dir)

    N = int(exp["N"])
    actions_space = exp["actions"]

    # env + coop function
    if exp["type"] == "matrix":
        env = MultiAgentMatrixGameEnv(
            payoff_table=exp["payoff"],
            actions=actions_space,
            num_agents=N,
            seed=seed,
        )
        coop_action = exp["coop_action"]
        coop_fn = lambda joint_actions, _state, _extra: coop_flag_matrix(joint_actions, coop_action)

    elif exp["type"] == "resource":
        env = ResourceSharingEnv(
            num_agents=N,
            max_stock=exp["max_stock"],
            regen=exp["regen"],
            penalty=exp["penalty"],
            seed=seed,
        )
        sustainable_threshold = exp["sustainable_threshold"]
        coop_action = None
        coop_fn = lambda _joint_actions, _state, extra: coop_flag_resource(extra, sustainable_threshold)

    else:
        raise ValueError(f"Unknown experiment type: {exp['type']}")

    # build agents for each baseline
    if baseline == "empathy":
        per_agent_alpha = [float(empathy_alpha)] * N
        hetero_tag = ""
    elif baseline == "hetero_fixed":
        per_agent_alpha, hetero_tag = get_hetero_profile(N, seed)
        # put profile tag into run_dir to keep different profiles separated
        run_dir = os.path.join(LOG_ROOT, name, baseline, hetero_tag, f"seed_{seed}")
        ensure_dir(run_dir)
    else:
        per_agent_alpha = [1.0] * N
        hetero_tag = ""

    agents: List[QAgent] = []
    for i in range(N):
        if baseline == "learn_alpha":
            agents.append(
                QAgent(
                    name=f"A{i+1}",
                    actions=actions_space,
                    alpha=LR,
                    gamma=GAMMA,
                    eps_start=EPS_START,
                    eps_end=EPS_END,
                    eps_decay=EPS_DECAY,
                    seed=seed + i,
                    empathy_alpha=1.0,            # initial value (won't matter much)
                    empathy_mode="bandit",
                    alpha_candidates=ALPHA_CANDIDATES,
                    alpha_bandit_eps=ALPHA_BANDIT_EPS,
                )
            )
        else:
            agents.append(
                QAgent(
                    name=f"A{i+1}",
                    actions=actions_space,
                    alpha=LR,
                    gamma=GAMMA,
                    eps_start=EPS_START,
                    eps_end=EPS_END,
                    eps_decay=EPS_DECAY,
                    seed=seed + i,
                    empathy_alpha=per_agent_alpha[i],
                    empathy_mode="fixed",
                )
            )

    csv_path = os.path.join(run_dir, "per_step.csv")

    # per-step fields (+ alpha per agent)
    fieldnames = (
    ["step", "experiment", "type", "N", "baseline", "empathy_alpha", "hetero_tag", "learn_alpha_signal"]
    + [f"alpha{i+1}" for i in range(N)]
    + ["alpha_mean"]
    + [f"a{i+1}" for i in range(N)]
    + [f"r{i+1}" for i in range(N)]
    + ["total_reward"]
    + ["coop_strict", "coop_fraction", "ineq_var", "stock_level", "collapse"]
    + ["ma_coop_strict", "ma_coop_fraction", "ma_total_reward", "ma_ineq_var", "ma_stock_level", "ma_collapse"]
    + ["ma_alpha_mean"]
)

    coop_window = deque(maxlen=EVAL_WINDOW)
    coopfrac_window = deque(maxlen=EVAL_WINDOW)
    reward_window = deque(maxlen=EVAL_WINDOW)
    ineq_window = deque(maxlen=EVAL_WINDOW)
    stock_window = deque(maxlen=EVAL_WINDOW)
    collapse_window = deque(maxlen=EVAL_WINDOW)
    alpha_mean_window = deque(maxlen=EVAL_WINDOW)

    state = env.reset()
    time_to_threshold: Optional[int] = None
    block_sig_sum = [0.0 for _ in range(N)]
    block_sig_count = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for step in range(STEPS):
            # 1) choose empathy alpha for this step (fixed or bandit)
            if baseline != "random":
                if baseline == "learn_alpha":
                    # pick new alpha only at block boundaries
                    if step % ALPHA_BLOCK == 0:
                        for ag in agents:
                            ag.select_empathy_alpha()
                    # else: keep ag.current_alpha as-is for this whole block
                else:
                    # fixed alpha baselines (calling each step is fine; it doesn't change)
                    for ag in agents:
                        ag.select_empathy_alpha()

            # 2) choose actions
            if baseline == "random":
                joint_actions = [random.choice(actions_space) for _ in range(N)]
            else:
                joint_actions = [ag.select_action(state) for ag in agents]

            rewards, next_state, extra = env.step(joint_actions)
            total_reward = float(sum(rewards))
            
            # 3) (learn_alpha) accumulate bandit signal over this block
            if baseline == "learn_alpha":
                block_sig_count += 1
                for i, ag in enumerate(agents):
                    sig = learn_alpha_signal(
                        mode=LEARN_ALPHA_SIGNAL,
                        rewards=[float(r) for r in rewards],
                        total_reward=total_reward,
                        N=N,
                        agent_alpha=ag.current_alpha,
                        i=i,
                    )
                    block_sig_sum[i] += float(sig)

                # update the bandit at the END of each block using block-average signal
                if (step + 1) % ALPHA_BLOCK == 0:
                    for i, ag in enumerate(agents):
                        avg_sig = block_sig_sum[i] / float(block_sig_count)
                        ag.update_alpha_bandit(avg_sig)
                    # reset for next block
                    block_sig_sum = [0.0 for _ in range(N)]
                    block_sig_count = 0

            # 4) learning update
            if baseline != "random":
                for i, ag in enumerate(agents):
                    if baseline == "team":
                        team_avg = total_reward / float(N)
                        # make shaping equal team_avg regardless of alpha
                        ag.current_alpha = 1.0
                        ag.update(state, joint_actions[i], team_avg, team_avg, next_state)
                    else:
                        r_self = float(rewards[i])
                        r_others_avg = (total_reward - r_self) / float(N - 1) if N > 1 else 0.0
                        ag.update(state, joint_actions[i], r_self, r_others_avg, next_state)

                    ag.decay_epsilon()

            # 5) metrics
            coop_strict = int(coop_fn(joint_actions, next_state, extra))
            alpha_mean = float(sum(ag.current_alpha for ag in agents)) / float(N) if baseline != "random" else float("nan")
            if baseline != "random":
                alpha_mean_window.append(alpha_mean)

            if exp["type"] == "matrix":
                coop_fraction = sum(1 for a in joint_actions if a == coop_action) / float(N)
                stock_level = ""
                collapse = ""
            else:
                coop_fraction = ""
                stock_level_val = float(extra.get("stock_level", 0.0))
                stock_level = stock_level_val
                collapse = 1 if stock_level_val <= 0.0 else 0

            mean_r = total_reward / float(N)
            ineq_var = float(sum((float(r) - mean_r) ** 2 for r in rewards) / float(N))

            coop_window.append(coop_strict)
            reward_window.append(total_reward)
            ineq_window.append(ineq_var)

            if exp["type"] == "matrix":
                coopfrac_window.append(float(coop_fraction))
            else:
                stock_window.append(float(stock_level))
                collapse_window.append(int(collapse))

            # moving averages
            if len(coop_window) == EVAL_WINDOW:
                ma_coop_strict = moving_average(coop_window)
                ma_total_reward = moving_average(reward_window)
                ma_ineq_var = moving_average(ineq_window)
                ma_alpha_mean = moving_average(alpha_mean_window) if baseline != "random" else ""

                if exp["type"] == "matrix":
                    ma_coop_fraction = moving_average(coopfrac_window)
                    ma_stock_level = ""
                    ma_collapse = ""
                else:
                    ma_coop_fraction = ""
                    ma_alpha_mean = moving_average(alpha_mean_window) if baseline != "random" else ""
                    ma_stock_level = moving_average(stock_window)
                    ma_collapse = moving_average(collapse_window)

                if time_to_threshold is None and ma_coop_strict >= COOP_THRESHOLD:
                    time_to_threshold = step
            else:
                ma_coop_strict = ""
                ma_coop_fraction = ""
                ma_total_reward = ""
                ma_ineq_var = ""
                ma_stock_level = ""
                ma_collapse = ""
                ma_alpha_mean = ""

            row = {
                "step": step,
                "experiment": name,
                "type": exp["type"],
                "N": N,
                "baseline": baseline,
                "empathy_alpha": empathy_alpha,
                "hetero_tag": hetero_tag,
                "learn_alpha_signal": (LEARN_ALPHA_SIGNAL if baseline == "learn_alpha" else ""),
                **{f"alpha{i+1}": round(float(agents[i].current_alpha), 3) if baseline != "random" else "" for i in range(N)},
                **{f"a{i+1}": joint_actions[i] for i in range(N)},
                **{f"r{i+1}": round(float(rewards[i]), 4) for i in range(N)},
                "total_reward": round(total_reward, 4),
                "coop_strict": int(coop_strict),
                "coop_fraction": (round(float(coop_fraction), 4) if exp["type"] == "matrix" else ""),
                "ineq_var": round(ineq_var, 6),
                "stock_level": (round(float(stock_level), 4) if exp["type"] == "resource" else ""),
                "collapse": (int(collapse) if exp["type"] == "resource" else ""),
                "ma_coop_strict": (round(float(ma_coop_strict), 4) if ma_coop_strict != "" else ""),
                "ma_coop_fraction": (round(float(ma_coop_fraction), 4) if ma_coop_fraction != "" else ""),
                "ma_total_reward": (round(float(ma_total_reward), 4) if ma_total_reward != "" else ""),
                "ma_ineq_var": (round(float(ma_ineq_var), 6) if ma_ineq_var != "" else ""),
                "ma_stock_level": (round(float(ma_stock_level), 4) if ma_stock_level != "" else ""),
                "ma_collapse": (round(float(ma_collapse), 4) if ma_collapse != "" else ""),
                "alpha_mean": (round(alpha_mean, 4) if baseline != "random" else ""),
                "ma_alpha_mean": (round(float(ma_alpha_mean), 4) if ma_alpha_mean != "" else ""),
            }
            w.writerow(row)

            state = next_state

    return csv_path, time_to_threshold


# -----------------------------
# main
# -----------------------------
def main() -> None:
    random.seed(SEED)
    ensure_dir(LOG_ROOT)

    condition_summary_rows: List[Dict[str, Any]] = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        exp_out_dir = os.path.join(LOG_ROOT, name)
        ensure_dir(exp_out_dir)

        # condition label -> list of per-seed per_step.csv paths
        cond_paths: Dict[str, List[str]] = defaultdict(list)

        for baseline in BASELINES:
            if baseline == "empathy":
                alphas = EMPATHY_ALPHAS
            else:
                alphas = [1.0]  # placeholder (not used)

            for alpha in alphas:
                # label logic
                if baseline == "empathy":
                    label = f"empathy α={alpha:.2f}"
                elif baseline == "learn_alpha":
                    label = f"learn_alpha ({LEARN_ALPHA_SIGNAL})"
                else:
                    label = f"{baseline}"

                ttc_vals: List[float] = []
                for seed in SEEDS:
                    csv_path, ttc = run_one_experiment(exp, alpha, seed, baseline)
                    cond_paths[label].append(csv_path)
                    if ttc is not None:
                        ttc_vals.append(float(ttc))

                condition_summary_rows.append({
                    "experiment": name,
                    "type": exp["type"],
                    "N": exp["N"],
                    "condition": label,
                    "n_seeds": len(SEEDS),
                    "ttc_mean": (round(sum(ttc_vals) / len(ttc_vals), 3) if ttc_vals else ""),
                    "ttc_std": (round(stdev(ttc_vals), 3) if ttc_vals else ""),
                })

        # plots: mean + 95% CI
        plot_metric_with_ci(
            cond_paths,
            exp_out_dir,
            title=f"{name}: Strict cooperation (moving avg)",
            field="ma_coop_strict",
            ylabel=f"MA coop_strict (window={EVAL_WINDOW})",
            out_name="ma_coop_strict_mean_ci.png",
        )

        plot_metric_with_ci(
            cond_paths,
            exp_out_dir,
            title=f"{name}: Total reward (moving avg)",
            field="ma_total_reward",
            ylabel=f"MA total reward (window={EVAL_WINDOW})",
            out_name="ma_total_reward_mean_ci.png",
        )

        plot_metric_with_ci(
            cond_paths,
            exp_out_dir,
            title=f"{name}: Inequality (reward variance, moving avg)",
            field="ma_ineq_var",
            ylabel=f"MA reward variance (window={EVAL_WINDOW})",
            out_name="ma_ineq_var_mean_ci.png",
        )

        plot_metric_with_ci(
            cond_paths,
            exp_out_dir,
            title=f"{name}: Mean empathy alpha (moving avg)",
            field="ma_alpha_mean",
            ylabel=f"MA alpha_mean (window={EVAL_WINDOW})",
            out_name="ma_alpha_mean_ci.png",
        )

        if exp["type"] == "matrix":
            plot_metric_with_ci(
                cond_paths,
                exp_out_dir,
                title=f"{name}: Cooperation fraction (moving avg)",
                field="ma_coop_fraction",
                ylabel=f"MA coop fraction (window={EVAL_WINDOW})",
                out_name="ma_coop_fraction_mean_ci.png",
            )

        if exp["type"] == "resource":
            plot_metric_with_ci(
                cond_paths,
                exp_out_dir,
                title=f"{name}: Stock level (moving avg)",
                field="ma_stock_level",
                ylabel=f"MA stock level (window={EVAL_WINDOW})",
                out_name="ma_stock_level_mean_ci.png",
            )
            plot_metric_with_ci(
                cond_paths,
                exp_out_dir,
                title=f"{name}: Collapse frequency (moving avg)",
                field="ma_collapse",
                ylabel=f"MA collapse freq (window={EVAL_WINDOW})",
                out_name="ma_collapse_mean_ci.png",
            )

        print(f"Done: {name} -> {os.path.abspath(exp_out_dir)}")

    # global summary
    summary_path = os.path.join(LOG_ROOT, "summary_conditions.csv")
    write_rows_csv(summary_path, condition_summary_rows)
    print(f"\nALL DONE. Condition summary: {os.path.abspath(summary_path)}")
    print(f"Logs root: {os.path.abspath(LOG_ROOT)}")


if __name__ == "__main__":
    main()