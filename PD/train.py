import os
import csv
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple, Deque, Optional

from config import (
    PD_PAYOFFS,
    ACTIONS,
    NUM_AGENTS_LIST,
    ALPHA,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    EPISODES,
    EVAL_WINDOW,
    SEED,
    ALPHAS,
    LOG_DIR,
    COOP_THRESHOLD,
)

from environment import MultiAgentMatrixGameEnv
from agent import QAgent

random.seed(SEED)


def ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def moving_average(window_vals: Deque[float]) -> float:
    return sum(window_vals) / len(window_vals)


def run_experiment(num_agents: int, empathy_alpha: float) -> Tuple[str, Optional[int]]:
    ensure_log_dir()

    env = MultiAgentMatrixGameEnv(
        payoff_table=PD_PAYOFFS,
        actions=ACTIONS,
        num_agents=num_agents,
        seed=SEED,
    )

    agents: List[QAgent] = [
        QAgent(
            name=f"A{i+1}",
            actions=ACTIONS,
            alpha=ALPHA,
            gamma=GAMMA,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY,
            seed=SEED + i,
            empathy_alpha=empathy_alpha,
        )
        for i in range(num_agents)
    ]

    csv_path = os.path.join(LOG_DIR, f"per_episode_alpha_{empathy_alpha:.2f}_N{num_agents}.csv")

    fieldnames = (
        ["episode", "alpha_emp", "N"]
        + [f"A{i+1}_action" for i in range(num_agents)]
        + [f"r{i+1}" for i in range(num_agents)]
        + [f"r{i+1}_shaped" for i in range(num_agents)]
        + [f"epsilon_A{i+1}" for i in range(num_agents)]
        + ["total_reward", "coop_flag", "moving_avg_coop", "moving_avg_total_reward"]
    )

    coop_window = deque(maxlen=EVAL_WINDOW)
    total_reward_window = deque(maxlen=EVAL_WINDOW)
    time_to_threshold: Optional[int] = None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        state = env.reset()

        for ep in range(1, EPISODES + 1):
            joint_actions = [ag.act(state) for ag in agents]
            rewards, next_state = env.step(joint_actions)

            total_reward = sum(rewards)

            shaped_rewards = []
            epsilons = []

            for i, ag in enumerate(agents):
                r_self = rewards[i]
                r_others_avg = (total_reward - r_self) / (num_agents - 1)

                shaped = ag.empathy_alpha * r_self + (1 - ag.empathy_alpha) * r_others_avg
                shaped_rewards.append(shaped)

                ag.update(state, joint_actions[i], r_self, r_others_avg, next_state)
                ag.decay_epsilon()
                epsilons.append(ag.eps)

            coop_flag = 1 if all(a == "C" for a in joint_actions) else 0
            coop_window.append(coop_flag)
            total_reward_window.append(total_reward)

            if len(coop_window) == EVAL_WINDOW:
                ma_coop = moving_average(coop_window)
                ma_total = moving_average(total_reward_window)
                if time_to_threshold is None and ma_coop >= COOP_THRESHOLD:
                    time_to_threshold = ep
            else:
                ma_coop = ""
                ma_total = ""

            row = [
                ep,
                empathy_alpha,
                num_agents,
                *joint_actions,
                *[float(f"{r:.4f}") for r in rewards],
                *[float(f"{r:.4f}") for r in shaped_rewards],
                *[float(f"{e:.4f}") for e in epsilons],
                float(f"{total_reward:.4f}"),
                coop_flag,
                ma_coop if ma_coop == "" else float(f"{ma_coop:.4f}"),
                ma_total if ma_total == "" else float(f"{ma_total:.4f}"),
            ]
            writer.writerow(row)

            state = next_state

    return csv_path, time_to_threshold


def compute_last_window_averages(csv_path: str) -> Tuple[Optional[float], Optional[float]]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    tail = [r for r in rows if r["moving_avg_coop"] != ""][-EVAL_WINDOW:]
    if not tail:
        return None, None

    avg_coop = sum(float(r["moving_avg_coop"]) for r in tail) / len(tail)
    avg_total = sum(float(r["moving_avg_total_reward"]) for r in tail) / len(tail)
    return avg_coop, avg_total


def write_summary(num_agents: int, summary_rows: List[List]) -> str:
    ensure_log_dir()
    summary_path = os.path.join(LOG_DIR, f"summary_N{num_agents}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "alpha_emp",
                "episodes",
                f"last_{EVAL_WINDOW}_avg_coop",
                f"last_{EVAL_WINDOW}_avg_total_reward",
                "time_to_coop_threshold",
            ]
        )
        writer.writerows(summary_rows)
    return summary_path


def plot_trends(num_agents: int) -> Tuple[str, str]:
    ensure_log_dir()

    # Cooperation plot
    plt.figure()
    for emp_alpha in ALPHAS:
        csv_path = os.path.join(LOG_DIR, f"per_episode_alpha_{emp_alpha:.2f}_N{num_agents}.csv")
        episodes, ma_coop = [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["moving_avg_coop"] == "":
                    continue
                episodes.append(int(row["episode"]))
                ma_coop.append(float(row["moving_avg_coop"]))
        if episodes:
            plt.plot(episodes, ma_coop, label=f"α={emp_alpha}")

    plt.xlabel("Episode")
    plt.ylabel(f"Moving avg. cooperation (all {num_agents} cooperate)")
    plt.title(f"Cooperation vs Episodes (N={num_agents})")
    plt.legend()
    plt.grid(True)
    coop_plot_path = os.path.join(LOG_DIR, f"coop_trends_N{num_agents}.png")
    plt.savefig(coop_plot_path, dpi=300)

    # Total reward plot
    plt.figure()
    for emp_alpha in ALPHAS:
        csv_path = os.path.join(LOG_DIR, f"per_episode_alpha_{emp_alpha:.2f}_N{num_agents}.csv")
        episodes, ma_total = [], []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["moving_avg_total_reward"] == "":
                    continue
                episodes.append(int(row["episode"]))
                ma_total.append(float(row["moving_avg_total_reward"]))
        if episodes:
            plt.plot(episodes, ma_total, label=f"α={emp_alpha}")

    plt.xlabel("Episode")
    plt.ylabel("Moving avg. total reward")
    plt.title(f"Total reward vs Episodes (N={num_agents})")
    plt.legend()
    plt.grid(True)
    total_plot_path = os.path.join(LOG_DIR, f"total_reward_trends_N{num_agents}.png")
    plt.savefig(total_plot_path, dpi=300)

    return coop_plot_path, total_plot_path


if __name__ == "__main__":
    ensure_log_dir()

    for N in NUM_AGENTS_LIST:
        summary_rows: List[List] = []

        for emp_alpha in ALPHAS:
            per_ep_csv, ttc = run_experiment(N, emp_alpha)
            last_coop, last_total = compute_last_window_averages(per_ep_csv)
            summary_rows.append([N, emp_alpha, EPISODES, last_coop, last_total, ttc])

        summary_csv = write_summary(N, summary_rows)
        coop_plot, total_plot = plot_trends(N)

        print(f"\n=== Done for N={N} ===")
        print(f"Per-episode logs: {os.path.abspath(LOG_DIR)}/per_episode_alpha_*_N{N}.csv")
        print(f"Summary:          {os.path.abspath(summary_csv)}")
        print(f"Coop plot:        {os.path.abspath(coop_plot)}")
        print(f"Total plot:       {os.path.abspath(total_plot)}")