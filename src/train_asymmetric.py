"""
Asymmetric Empathy Training

Train agents with DIFFERENT empathy levels to study:
- How empathic agents interact with selfish agents
- Exploitation vs cooperation dynamics
- Emergent behavior in mixed populations
"""

import os
import csv
import random
from collections import deque
from itertools import product

from config import (
    PD_PAYOFFS, ACTIONS, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY,
    EPISODES, EVAL_WINDOW, SEED, LOG_DIR, COOP_THRESHOLD
)
from environment import MatrixGameEnv
from agent import QAgent

random.seed(SEED)


def moving_average(window_vals):
    return sum(window_vals) / len(window_vals)


def run_asymmetric_experiment(alpha1: float, alpha2: float):
    """
    Run experiment with two agents having different empathy levels.
    
    Args:
        alpha1: Empathy level for agent 1 (1.0 = selfish)
        alpha2: Empathy level for agent 2 (1.0 = selfish)
    """
    env = MatrixGameEnv(PD_PAYOFFS, ACTIONS, seed=SEED)
    
    # Create agents with DIFFERENT empathy levels
    a1 = QAgent("A", ACTIONS, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY, 
                seed=SEED, empathy_alpha=alpha1)
    a2 = QAgent("B", ACTIONS, ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY, 
                seed=SEED+1, empathy_alpha=alpha2)

    state = env.reset()

    coop_flag_win = deque(maxlen=EVAL_WINDOW)
    total_reward_win = deque(maxlen=EVAL_WINDOW)
    r1_win = deque(maxlen=EVAL_WINDOW)
    r2_win = deque(maxlen=EVAL_WINDOW)

    # Logging per-episode CSV
    os.makedirs(LOG_DIR, exist_ok=True)
    per_ep_path = os.path.join(LOG_DIR, f"asymmetric_a1_{alpha1:.2f}_a2_{alpha2:.2f}.csv")
    
    # For regret calculation
    optimal_welfare = 6.0  # (C, C) optimal outcome
    cumulative_social_regret = 0.0
    cumulative_regret_a1 = 0.0
    cumulative_regret_a2 = 0.0
    
    with open(per_ep_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "alpha_a1", "alpha_a2", "A_action", "B_action",
            "r1", "r2", "r1_shaped", "r2_shaped", "total_reward",
            "epsilon_A", "epsilon_B", "coop_flag",
            "moving_avg_coop", "moving_avg_total_reward",
            "moving_avg_r1", "moving_avg_r2",
            "social_regret", "cumulative_social_regret", "avg_social_regret",
            "regret_a1", "cumulative_regret_a1", "avg_regret_a1",
            "regret_a2", "cumulative_regret_a2", "avg_regret_a2"
        ])

        time_to_threshold = None
        for ep in range(1, EPISODES + 1):
            act1 = a1.select_action(state)
            act2 = a2.select_action(state)

            (r1, r2), next_state = env.step(act1, act2)

            # Shaped rewards actually used for learning
            r1_shaped = a1.empathy_alpha * r1 + (1 - a1.empathy_alpha) * r2
            r2_shaped = a2.empathy_alpha * r2 + (1 - a2.empathy_alpha) * r1

            a1.update(state, act1, r1, r2, next_state)
            a2.update(state, act2, r2, r1, next_state)

            a1.decay_epsilon()
            a2.decay_epsilon()

            coop_flag = 1 if (act1 == "C" and act2 == "C") else 0
            coop_flag_win.append(coop_flag)
            total_reward = r1 + r2
            total_reward_win.append(total_reward)
            r1_win.append(r1)
            r2_win.append(r2)

            # Social Regret: vs optimal social welfare
            social_regret = optimal_welfare - total_reward
            cumulative_social_regret += social_regret
            avg_social_regret = cumulative_social_regret / ep
            
            # Individual Regret: vs best response given other's action
            if act2 == "C":
                best_r1 = PD_PAYOFFS[("D", "C")][0]
            else:
                best_r1 = PD_PAYOFFS[("D", "D")][0]
            regret_a1 = best_r1 - r1
            cumulative_regret_a1 += regret_a1
            avg_regret_a1 = cumulative_regret_a1 / ep
            
            if act1 == "C":
                best_r2 = PD_PAYOFFS[("C", "D")][1]
            else:
                best_r2 = PD_PAYOFFS[("D", "D")][1]
            regret_a2 = best_r2 - r2
            cumulative_regret_a2 += regret_a2
            avg_regret_a2 = cumulative_regret_a2 / ep

            ma_coop = moving_average(coop_flag_win) if len(coop_flag_win) == EVAL_WINDOW else ""
            ma_total = moving_average(total_reward_win) if len(total_reward_win) == EVAL_WINDOW else ""
            ma_r1 = moving_average(r1_win) if len(r1_win) == EVAL_WINDOW else ""
            ma_r2 = moving_average(r2_win) if len(r2_win) == EVAL_WINDOW else ""

            if time_to_threshold is None and isinstance(ma_coop, float) and ma_coop >= COOP_THRESHOLD:
                time_to_threshold = ep

            writer.writerow([
                ep, alpha1, alpha2, act1, act2, r1, r2, r1_shaped, r2_shaped,
                total_reward, a1.eps, a2.eps, coop_flag, ma_coop, ma_total,
                ma_r1, ma_r2,
                social_regret, cumulative_social_regret, avg_social_regret,
                regret_a1, cumulative_regret_a1, avg_regret_a1,
                regret_a2, cumulative_regret_a2, avg_regret_a2
            ])

            state = next_state

    return per_ep_path, time_to_threshold


def write_asymmetric_summary(rows):
    """Write summary of asymmetric experiments."""
    os.makedirs(LOG_DIR, exist_ok=True)
    summary_path = os.path.join(LOG_DIR, "asymmetric_summary.csv")
    
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alpha_a1", "alpha_a2", "episodes", 
            f"last_{EVAL_WINDOW}_avg_coop",
            f"last_{EVAL_WINDOW}_avg_total_reward",
            f"last_{EVAL_WINDOW}_avg_r1",
            f"last_{EVAL_WINDOW}_avg_r2",
            "time_to_coop_threshold"
        ])
        for row in rows:
            writer.writerow(row)
    return summary_path


def compute_last_window_averages(csv_path):
    """Compute averages for last evaluation window."""
    last_rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            last_rows.append(r)
    
    tail = [r for r in last_rows if r["moving_avg_coop"] != ""][-EVAL_WINDOW:]
    if not tail:
        return None, None, None, None
    
    avg_coop = sum(float(r["moving_avg_coop"]) for r in tail) / len(tail)
    avg_total = sum(float(r["moving_avg_total_reward"]) for r in tail) / len(tail)
    avg_r1 = sum(float(r["moving_avg_r1"]) for r in tail) / len(tail)
    avg_r2 = sum(float(r["moving_avg_r2"]) for r in tail) / len(tail)
    
    return avg_coop, avg_total, avg_r1, avg_r2


if __name__ == "__main__":
    # Define interesting asymmetric combinations
    asymmetric_pairs = [
        # (alpha_agent1, alpha_agent2)
        (1.0, 1.0),   # Both selfish (baseline)
        (1.0, 0.5),   # Selfish vs Empathic
        (1.0, 0.2),   # Selfish vs Very Empathic
        (0.5, 1.0),   # Empathic vs Selfish (reversed)
        (0.2, 1.0),   # Very Empathic vs Selfish (reversed)
        (0.5, 0.5),   # Both moderately empathic
        (0.2, 0.2),   # Both very empathic
        (0.8, 0.2),   # Slightly selfish vs Very empathic
        (0.2, 0.8),   # Very empathic vs Slightly selfish
    ]
    
    print("\n" + "="*70)
    print("ASYMMETRIC EMPATHY EXPERIMENTS")
    print("="*70 + "\n")
    
    summary_rows = []
    
    for alpha1, alpha2 in asymmetric_pairs:
        print(f"Training: Agent 1 (α={alpha1:.2f}) vs Agent 2 (α={alpha2:.2f})...")
        
        per_ep_csv, ttc = run_asymmetric_experiment(alpha1, alpha2)
        avg_coop, avg_total, avg_r1, avg_r2 = compute_last_window_averages(per_ep_csv)
        
        summary_rows.append([alpha1, alpha2, EPISODES, avg_coop, avg_total, 
                           avg_r1, avg_r2, ttc])
        
        print(f"  Cooperation: {avg_coop*100:.1f}%")
        print(f"  Agent 1 avg reward: {avg_r1:.2f}")
        print(f"  Agent 2 avg reward: {avg_r2:.2f}")
        print(f"  Total welfare: {avg_total:.2f}\n")

    summary_csv = write_asymmetric_summary(summary_rows)
    
    print("="*70)
    print("Asymmetric training complete!")
    print(f"Summary: {os.path.abspath(summary_csv)}")
    print("="*70)
