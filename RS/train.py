# train_resource.py

import os
import csv
import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt

from agent import QAgent
from environment import ResourceSharingEnv
from config import (
    NUM_AGENTS,
    ACTIONS,
    MAX_STOCK,
    REGEN,
    PENALTY,
    SEED,
    LR,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    EPISODES,
    EVAL_WINDOW,
    EMPATHY_ALPHAS,
    LOG_DIR,
    SUSTAINABLE_THRESHOLD,
    COOP_THRESHOLD,
)


def ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def moving_average(vals: deque) -> float:
    return sum(vals) / len(vals)


def run_experiment(empathy_alpha: float) -> Tuple[str, int | None]:
    ensure_log_dir()

    env = ResourceSharingEnv(
        num_agents=NUM_AGENTS,
        max_stock=MAX_STOCK,
        regen=REGEN,
        penalty=PENALTY,
        seed=SEED,
    )

    agents: List[QAgent] = [
        QAgent(
            name=f"A{i+1}",
            actions=ACTIONS,
            alpha=LR,
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
        LOG_DIR, f"per_episode_empathy_{empathy_alpha:.2f}_N{NUM_AGENTS}.csv"
    )

    fieldnames = (
        ["episode", "empathy_alpha"]
        + [f"A{i+1}_action" for i in range(NUM_AGENTS)]
        + [f"r{i+1}" for i in range(NUM_AGENTS)]
        + [f"r{i+1}_shaped" for i in range(NUM_AGENTS)]
        + [f"epsilon_A{i+1}" for i in range(NUM_AGENTS)]
        + [
            "total_extraction",
            "stock_level",
            "total_reward",
            "coop_flag",
            "moving_avg_coop",
            "moving_avg_total_reward",
        ]
    )

    coop_window = deque(maxlen=EVAL_WINDOW)
    reward_window = deque(maxlen=EVAL_WINDOW)
    time_to_threshold = None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        state = env.reset()

        for ep in range(1, EPISODES + 1):
            joint_actions = [ag.act(state) for ag in agents]

            rewards, next_state = env.step(joint_actions)
            total_reward = float(sum(rewards))
            total_extraction = int(sum(joint_actions))
            stock_level = int(next_state)

            shaped_rewards = []
            epsilons = []

            for i, ag in enumerate(agents):
                r_self = float(rewards[i])
                r_others_avg = (total_reward - r_self) / (NUM_AGENTS - 1)

                shaped = ag.empathy_alpha * r_self + (1 - ag.empathy_alpha) * r_others_avg
                shaped_rewards.append(shaped)

                ag.update(state, joint_actions[i], r_self, r_others_avg, next_state)
                ag.decay_epsilon()
                epsilons.append(float(ag.eps))

            # "cooperation" = sustainable total extraction
            coop_flag = 1 if total_extraction <= SUSTAINABLE_THRESHOLD else 0

            coop_window.append(coop_flag)
            reward_window.append(total_reward)

            if len(coop_window) == EVAL_WINDOW:
                ma_coop = moving_average(coop_window)
                ma_reward = moving_average(reward_window)
                if time_to_threshold is None and ma_coop >= COOP_THRESHOLD:
                    time_to_threshold = ep
            else:
                ma_coop = ""
                ma_reward = ""

            writer.writerow([
                ep,
                empathy_alpha,
                *joint_actions,
                *[round(float(r), 4) for r in rewards],
                *[round(float(r), 4) for r in shaped_rewards],
                *[round(float(e), 4) for e in epsilons],
                total_extraction,
                stock_level,
                round(total_reward, 4),
                coop_flag,
                ma_coop if ma_coop == "" else round(float(ma_coop), 4),
                ma_reward if ma_reward == "" else round(float(ma_reward), 4),
            ])

            state = next_state

    return csv_path, time_to_threshold


def compute_last_window_averages(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    tail = [r for r in rows if r["moving_avg_coop"] != ""][-EVAL_WINDOW:]
    if not tail:
        return None, None

    avg_coop = sum(float(r["moving_avg_coop"]) for r in tail) / len(tail)
    avg_total = sum(float(r["moving_avg_total_reward"]) for r in tail) / len(tail)
    return avg_coop, avg_total


def write_summary(summary_rows):
    ensure_log_dir()
    path = os.path.join(LOG_DIR, f"summary_resource_N{NUM_AGENTS}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "empathy_alpha",
            "episodes",
            "last_window_avg_coop",
            "last_window_avg_total_reward",
            "time_to_coop_threshold",
        ])
        writer.writerows(summary_rows)
    return path


def main():
    random.seed(SEED)

    summary_rows = []
    per_episode_paths = {}

    for emp in EMPATHY_ALPHAS:
        per_ep_csv, ttc = run_experiment(emp)
        last_coop, last_total = compute_last_window_averages(per_ep_csv)
        summary_rows.append([emp, EPISODES, last_coop, last_total, ttc])
        per_episode_paths[emp] = per_ep_csv

    summary_csv = write_summary(summary_rows)
    print("Logging complete:")
    print(f" - Summary: {os.path.abspath(summary_csv)}")
    print(f" - Logs dir: {os.path.abspath(LOG_DIR)}")

    # Plot: cooperation
    plt.figure()
    for emp in EMPATHY_ALPHAS:
        xs, ys = [], []
        with open(per_episode_paths[emp], newline="") as f:
            for row in csv.DictReader(f):
                if row["moving_avg_coop"] == "":
                    continue
                xs.append(int(row["episode"]))
                ys.append(float(row["moving_avg_coop"]))
        if xs:
            plt.plot(xs, ys, label=f"α={emp}")
    plt.xlabel("Episode")
    plt.ylabel("Moving avg. sustainable behavior")
    plt.title(f"Resource sharing: cooperation vs episodes (N={NUM_AGENTS})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, f"coop_trends_resource_N{NUM_AGENTS}.png"), dpi=300)

    # Plot: total reward
    plt.figure()
    for emp in EMPATHY_ALPHAS:
        xs, ys = [], []
        with open(per_episode_paths[emp], newline="") as f:
            for row in csv.DictReader(f):
                if row["moving_avg_total_reward"] == "":
                    continue
                xs.append(int(row["episode"]))
                ys.append(float(row["moving_avg_total_reward"]))
        if xs:
            plt.plot(xs, ys, label=f"α={emp}")
    plt.xlabel("Episode")
    plt.ylabel("Moving avg. total reward")
    plt.title(f"Resource sharing: total reward vs episodes (N={NUM_AGENTS})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, f"total_reward_trends_resource_N{NUM_AGENTS}.png"), dpi=300)


if __name__ == "__main__":
    main()