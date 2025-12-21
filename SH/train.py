# train.py

import os
import csv
import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt

from config import (
    ACTIONS,
    PAYOFF_MATRIX,
    NUM_AGENTS,
    ALPHA,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    EPISODES,
    EVAL_WINDOW,
    SEED,
    EMPATHY_ALPHAS,
    COOP_THRESHOLD,
    LOG_DIR,
)

from environment import MultiAgentMatrixGameEnv
from agent import QAgent


# ---------- UTILS ----------

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def moving_average(vals: deque) -> float:
    return sum(vals) / len(vals)


# ---------- SINGLE RUN ----------

def run_experiment(empathy_alpha: float) -> Tuple[str, int | None]:
    ensure_log_dir()

    env = MultiAgentMatrixGameEnv(
        payoff_table=PAYOFF_MATRIX,
        actions=ACTIONS,
        num_agents=NUM_AGENTS,
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
        for i in range(NUM_AGENTS)
    ]

    csv_path = os.path.join(
        LOG_DIR, f"per_episode_alpha_{empathy_alpha:.2f}_N{NUM_AGENTS}.csv"
    )

    fieldnames = (
        ["episode", "alpha_emp"]
        + [f"A{i+1}_action" for i in range(NUM_AGENTS)]
        + [f"r{i+1}" for i in range(NUM_AGENTS)]
        + [f"r{i+1}_shaped" for i in range(NUM_AGENTS)]
        + [f"epsilon_A{i+1}" for i in range(NUM_AGENTS)]
        + ["total_reward", "coop_flag", "moving_avg_coop", "moving_avg_total_reward"]
    )

    coop_window = deque(maxlen=EVAL_WINDOW)
    reward_window = deque(maxlen=EVAL_WINDOW)
    time_to_threshold = None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        state = env.reset()

        for ep in range(1, EPISODES + 1):
            actions = [ag.act(state) for ag in agents]

            rewards, next_state = env.step(actions)
            total_reward = sum(rewards)

            shaped_rewards = []
            epsilons = []

            for i, ag in enumerate(agents):
                r_self = rewards[i]
                r_others_avg = (total_reward - r_self) / (NUM_AGENTS - 1)

                shaped = (
                    ag.empathy_alpha * r_self
                    + (1 - ag.empathy_alpha) * r_others_avg
                )
                shaped_rewards.append(shaped)

                ag.update(state, actions[i], r_self, r_others_avg, next_state)
                ag.decay_epsilon()
                epsilons.append(ag.eps)

            # Stag Hunt cooperation = everyone plays "S"
            coop_flag = 1 if all(a == "S" for a in actions) else 0

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
                *actions,
                *[round(r, 4) for r in rewards],
                *[round(r, 4) for r in shaped_rewards],
                *[round(e, 4) for e in epsilons],
                round(total_reward, 4),
                coop_flag,
                ma_coop if ma_coop == "" else round(ma_coop, 4),
                ma_reward if ma_reward == "" else round(ma_reward, 4),
            ])

            state = next_state

    return csv_path, time_to_threshold


# ---------- MAIN ----------

def main():
    random.seed(SEED)

    summary = {}
    paths = {}

    for alpha in EMPATHY_ALPHAS:
        path, ttc = run_experiment(alpha)
        summary[alpha] = ttc
        paths[alpha] = path

    print("Runs complete.")
    print(f"Logs in: {os.path.abspath(LOG_DIR)}")

if __name__ == "__main__":
    main()