# train_resource.py

import os
import csv
import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt

from agent import QAgent
from resource_environment import ResourceSharingEnv

# ---------- CONFIG ----------

NUM_AGENTS = 4

# Actions are extraction levels: 0 = low, 1 = medium, 2 = high
ACTIONS = [0, 1, 2]

# Q-learning hyperparameters
ALPHA = 0.2       # learning rate
GAMMA = 0.95      # discount factor
EPS_START = 1.0   # initial epsilon
EPS_END = 0.05    # minimum epsilon
EPS_DECAY = 0.995

EPISODES = 5000         # number of training episodes
EVAL_WINDOW = 100       # rolling window size for moving averages
SEED = 42

# Empathy levels to test (alpha = weight on own reward)
ALPHAS = [1.0, 0.8, 0.5, 0.2]

# logging
LOG_DIR = "logs_resource"

# "Cooperation" definition for resource sharing:
# here we say behaviour is "cooperative" if the total extraction is below
# some sustainable threshold in a given episode.
SUSTAINABLE_THRESHOLD = 4  # you can tune this

# threshold for "time to cooperation" metric
COOP_THRESHOLD = 0.8


# ---------- UTILS ----------

def ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def moving_average(window_vals: deque) -> float:
    return sum(window_vals) / len(window_vals)


# ---------- TRAINING FOR ONE EMPATHY LEVEL ----------

def run_experiment(empathy_alpha: float) -> Tuple[str, int]:
    """
    Run one training run for a given empathy level.

    Returns:
        (path_to_per_episode_csv, time_to_cooperation_threshold)
    """
    ensure_log_dir()

    env = ResourceSharingEnv(
        num_agents=NUM_AGENTS,
        max_stock=10,
        regen=2.0,
        penalty=-0.5,
        seed=SEED,
    )

    # all agents use the same empathy_alpha in this experiment
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
        for i in range(NUM_AGENTS)
    ]

    csv_path = os.path.join(
        LOG_DIR, f"per_episode_alpha_{empathy_alpha:.2f}_N{NUM_AGENTS}_resource.csv"
    )

    fieldnames = (
        ["episode", "alpha_emp"]
        + [f"A{i+1}_action" for i in range(NUM_AGENTS)]
        + [f"r{i+1}" for i in range(NUM_AGENTS)]
        + [f"r{i+1}_shaped" for i in range(NUM_AGENTS)]
        + [f"epsilon_A{i+1}" for i in range(NUM_AGENTS)]
        + ["total_extraction", "stock_level", "total_reward",
           "coop_flag", "moving_avg_coop", "moving_avg_total_reward"]
    )

    coop_window = deque(maxlen=EVAL_WINDOW)
    total_reward_window = deque(maxlen=EVAL_WINDOW)
    time_to_threshold = None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        state = env.reset()

        for ep in range(1, EPISODES + 1):
            # joint action selection
            actions = [ag.act(state) for ag in agents]

            # environment transition
            rewards, next_state = env.step(actions)
            total_reward = sum(rewards)
            total_extraction = sum(actions)
            stock_level = next_state  # integer state = current stock

            shaped_rewards = []
            epsilons = []

            # agent updates
            for i, ag in enumerate(agents):
                r_self = rewards[i]
                if NUM_AGENTS > 1:
                    r_others_avg = (total_reward - r_self) / (NUM_AGENTS - 1)
                else:
                    r_others_avg = 0.0

                shaped = (
                    ag.empathy_alpha * r_self
                    + (1 - ag.empathy_alpha) * r_others_avg
                )
                shaped_rewards.append(shaped)

                ag.update(state, actions[i], r_self, r_others_avg, next_state)
                ag.decay_epsilon()
                epsilons.append(ag.eps)

            # define "cooperation" here as sustainable usage
            coop_flag = 1 if total_extraction <= SUSTAINABLE_THRESHOLD else 0
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
                *actions,
                *[float(f"{r:.4f}") for r in rewards],
                *[float(f"{r:.4f}") for r in shaped_rewards],
                *[float(f"{e:.4f}") for e in epsilons],
                total_extraction,
                stock_level,
                float(f"{total_reward:.4f}"),
                coop_flag,
                ma_coop if ma_coop == "" else float(f"{ma_coop:.4f}"),
                ma_total if ma_total == "" else float(f"{ma_total:.4f}"),
            ]
            writer.writerow(row)

            state = next_state

    return csv_path, time_to_threshold


# ---------- SUMMARY COMPUTATION ----------

def compute_last_window_averages(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None, None

    tail = [r for r in rows if r["moving_avg_coop"] != ""][-EVAL_WINDOW:]
    if not tail:
        return None, None

    avg_coop = sum(float(r["moving_avg_coop"]) for r in tail) / len(tail)
    avg_total = sum(float(r["moving_avg_total_reward"]) for r in tail) / len(tail)
    return avg_coop, avg_total


def write_summary(summary_rows):
    ensure_log_dir()
    summary_path = os.path.join(LOG_DIR, f"summary_resource_N{NUM_AGENTS}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "alpha_emp",
                "episodes",
                "last_100_avg_coop",
                "last_100_avg_total_reward",
                "time_to_coop_threshold",
            ]
        )
        for row in summary_rows:
            writer.writerow(row)
    return summary_path


# ---------- MAIN: RUN EXPERIMENTS + PLOTS ----------

if __name__ == "__main__":
    random.seed(SEED)

    # 1) Run experiments for each empathy level
    summary_rows = []
    per_episode_paths = {}

    for emp_alpha in ALPHAS:
        per_ep_csv, ttc = run_experiment(emp_alpha)
        last_coop, last_total = compute_last_window_averages(per_ep_csv)
        summary_rows.append([emp_alpha, EPISODES, last_coop, last_total, ttc])
        per_episode_paths[emp_alpha] = per_ep_csv

    summary_csv = write_summary(summary_rows)
    print("Logging complete. Import CSVs into Excel if you like:")
    print(f" - Summary: {os.path.abspath(summary_csv)}")
    print(f" - Per-episode logs are in: {os.path.abspath(LOG_DIR)}")

    # 2) Plots

    # Cooperation plot
    plt.figure()
    for emp_alpha in ALPHAS:
        csv_path = per_episode_paths[emp_alpha]
        episodes = []
        ma_coop = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["moving_avg_coop"] == "":
                    continue
                episodes.append(int(row["episode"]))
                ma_coop.append(float(row["moving_avg_coop"]))

        if episodes:
            plt.plot(episodes, ma_coop, label=f"α = {emp_alpha}")

    plt.xlabel("Episode")
    plt.ylabel("Moving avg. sustainable behaviour (coop_flag)")
    plt.title(f"Resource sharing: 'cooperation' vs episodes (N = {NUM_AGENTS})")
    plt.legend()
    plt.grid(True)
    coop_plot_path = os.path.join(LOG_DIR, f"coop_trends_resource_N{NUM_AGENTS}.png")
    plt.savefig(coop_plot_path, dpi=300)
    print(f"Saved cooperation plot to: {os.path.abspath(coop_plot_path)}")

    # Total reward plot
    plt.figure()
    for emp_alpha in ALPHAS:
        csv_path = per_episode_paths[emp_alpha]
        episodes = []
        ma_total = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["moving_avg_total_reward"] == "":
                    continue
                episodes.append(int(row["episode"]))
                ma_total.append(float(row["moving_avg_total_reward"]))

        if episodes:
            plt.plot(episodes, ma_total, label=f"α = {emp_alpha}")

    plt.xlabel("Episode")
    plt.ylabel("Moving avg. total reward")
    plt.title(f"Resource sharing: total reward vs episodes (N = {NUM_AGENTS})")
    plt.legend()
    plt.grid(True)
    total_plot_path = os.path.join(
        LOG_DIR, f"total_reward_trends_resource_N{NUM_AGENTS}.png"
    )
    plt.savefig(total_plot_path, dpi=300)
    print(f"Saved total reward plot to: {os.path.abspath(total_plot_path)}")

    # If you want the plots to pop up in a window when running locally, uncomment:
    # plt.show()