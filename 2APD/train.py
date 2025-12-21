import os
import csv
import random
from collections import deque

from config import (
    PD_PAYOFFS, 
    ACTIONS,
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
    COOP_THRESHOLD
)
from environment import MatrixGameEnv
from agent import QAgent

random.seed(SEED)


def moving_average(window_vals):
    return sum(window_vals) / len(window_vals)


def run_experiment(empathy_alpha: float):
    env = MatrixGameEnv(PD_PAYOFFS, ACTIONS, seed=SEED)
    a1 = QAgent("A", ACTIONS, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY, seed=SEED,
                empathy_alpha=empathy_alpha)
    a2 = QAgent("B", ACTIONS, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY, seed=SEED+1,
                empathy_alpha=empathy_alpha)

    state = env.reset()

    coop_flag_win = deque(maxlen=EVAL_WINDOW)
    total_reward_win = deque(maxlen=EVAL_WINDOW)

    # Logging per-episode CSV
    os.makedirs(LOG_DIR, exist_ok=True)
    per_ep_path = os.path.join(LOG_DIR, f"per_episode_alpha_{empathy_alpha:.2f}.csv")
    with open(per_ep_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "alpha_emp", "A_action", "B_action",
            "r1", "r2", "r1_shaped", "r2_shaped", "total_reward",
            "epsilon_A", "epsilon_B", "coop_flag",
            "moving_avg_coop", "moving_avg_total_reward"
        ])

        time_to_threshold = None
        for ep in range(1, EPISODES + 1):
            act1 = a1.select_action(state)
            act2 = a2.select_action(state)

            (r1, r2), next_state = env.step(act1, act2)

            # shaped rewards actually used for learning
            r1_shaped = a1.empathy_alpha * r1 + (1 - a1.empathy_alpha) * r2
            r2_shaped = a2.empathy_alpha * r2 + (1 - a2.empathy_alpha) * r1

            a1.update(state, act1, r1, r2, next_state)
            a2.update(state, act2, r2, r1, next_state)

            a1.decay_epsilon(); a2.decay_epsilon()

            coop_flag = 1 if (act1 == "C" and act2 == "C") else 0
            coop_flag_win.append(coop_flag)
            total_reward = r1 + r2
            total_reward_win.append(total_reward)

            ma_coop = moving_average(coop_flag_win) if len(coop_flag_win) == EVAL_WINDOW else ""
            ma_total = moving_average(total_reward_win) if len(total_reward_win) == EVAL_WINDOW else ""

            if time_to_threshold is None and isinstance(ma_coop, float) and ma_coop >= COOP_THRESHOLD:
                time_to_threshold = ep

            writer.writerow([
                ep, empathy_alpha, act1, act2, r1, r2, r1_shaped, r2_shaped,
                total_reward, a1.eps, a2.eps, coop_flag, ma_coop, ma_total
            ])

            state = next_state

    return per_ep_path, time_to_threshold


def write_summary(rows):
    os.makedirs(LOG_DIR, exist_ok=True)
    summary_path = os.path.join(LOG_DIR, "summary_by_alpha.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha_emp", "episodes", f"last_{EVAL_WINDOW}_avg_coop",
            f"last_{EVAL_WINDOW}_avg_total_reward", "time_to_coop_threshold"
        ])
        for row in rows:
            writer.writerow(row)
    return summary_path


def compute_last_window_averages(csv_path):
    last_rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            last_rows.append(r)
    # take last EVAL_WINDOW rows that have moving averages filled
    tail = [r for r in last_rows if r["moving_avg_coop"] != ""][-EVAL_WINDOW:]
    if not tail:
        return None, None
    avg_coop = sum(float(r["moving_avg_coop"]) for r in tail) / len(tail)
    avg_total = sum(float(r["moving_avg_total_reward"]) for r in tail) / len(tail)
    return avg_coop, avg_total


if __name__ == "__main__":
    summary_rows = []
    for emp_alpha in ALPHAS:
        per_ep_csv, ttc = run_experiment(emp_alpha)
        last_coop, last_total = compute_last_window_averages(per_ep_csv)
        summary_rows.append([emp_alpha, EPISODES, last_coop, last_total, ttc])

    summary_csv = write_summary(summary_rows)
    print("Logging complete. Import CSVs into Excel:")
    print(f" - Per-episode logs: {os.path.abspath(LOG_DIR)}/per_episode_alpha_*.csv")
    print(f" - Summary: {os.path.abspath(summary_csv)}")