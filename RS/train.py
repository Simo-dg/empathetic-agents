import os
import csv
import random
from collections import deque
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt

from agent import QAgent
from environment import MultiAgentMatrixGameEnv, ResourceSharingEnv

from config import (
    RUN_GAMES,
    NUM_AGENTS,
    LR,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    EPISODES,
    EVAL_WINDOW,
    SEED,
    EMPATHY_ALPHAS,
    COOP_THRESHOLD,
    # stag
    STAG_ACTIONS,
    STAG_PAYOFFS,
    LOG_DIR_STAG,
    coop_flag_stag,
    # resource
    RS_ACTIONS,
    RS_MAX_STOCK,
    RS_REGEN,
    RS_PENALTY,
    LOG_DIR_RESOURCE,
    coop_flag_resource,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(vals: deque) -> float:
    return sum(vals) / len(vals)


def build_game(game_name: str):
    """
    Returns:
      env, actions, log_dir, coop_flag_fn, game_tag
    coop_flag_fn signature: (joint_actions, state, extra) -> 0/1
    """
    if game_name == "stag":
        env = MultiAgentMatrixGameEnv(
            payoff_table=STAG_PAYOFFS,
            actions=STAG_ACTIONS,
            num_agents=NUM_AGENTS,
            seed=SEED,
        )
        return env, STAG_ACTIONS, LOG_DIR_STAG, coop_flag_stag, "stag"

    if game_name == "resource":
        env = ResourceSharingEnv(
            num_agents=NUM_AGENTS,
            max_stock=RS_MAX_STOCK,
            regen=RS_REGEN,
            penalty=RS_PENALTY,
            seed=SEED,
        )
        return env, RS_ACTIONS, LOG_DIR_RESOURCE, coop_flag_resource, "resource"

    raise ValueError(f"Unknown game_name: {game_name}")


def run_experiment(game_name: str, empathy_alpha: float) -> Tuple[str, int | None]:
    env, actions_space, log_dir, coop_fn, tag = build_game(game_name)
    ensure_dir(log_dir)

    agents: List[QAgent] = [
        QAgent(
            name=f"A{i+1}",
            actions=actions_space,
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
        log_dir, f"per_step_{tag}_empathy_{empathy_alpha:.2f}_N{NUM_AGENTS}.csv"
    )

    # unified logging fields
    fieldnames = (
        ["step", "game", "empathy_alpha"]
        + [f"A{i+1}_action" for i in range(NUM_AGENTS)]
        + [f"r{i+1}" for i in range(NUM_AGENTS)]
        + [f"r{i+1}_shaped" for i in range(NUM_AGENTS)]
        + [f"epsilon_A{i+1}" for i in range(NUM_AGENTS)]
        + ["state", "total_reward", "coop_flag", "moving_avg_coop", "moving_avg_total_reward"]
        + ["extra_total_extraction", "extra_stock_level"]
    )

    coop_window = deque(maxlen=EVAL_WINDOW)
    reward_window = deque(maxlen=EVAL_WINDOW)
    time_to_threshold = None

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        state = env.reset()

        for step in range(1, EPISODES + 1):
            joint_actions = [ag.act(state) for ag in agents]
            rewards, next_state, extra = env.step(joint_actions)

            total_reward = float(sum(rewards))

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

            coop_flag = int(coop_fn(joint_actions, next_state, extra))
            coop_window.append(coop_flag)
            reward_window.append(total_reward)

            if len(coop_window) == EVAL_WINDOW:
                ma_coop = moving_average(coop_window)
                ma_reward = moving_average(reward_window)
                if time_to_threshold is None and ma_coop >= COOP_THRESHOLD:
                    time_to_threshold = step
            else:
                ma_coop = ""
                ma_reward = ""

            writer.writerow([
                step,
                tag,
                empathy_alpha,
                *joint_actions,
                *[round(float(r), 4) for r in rewards],
                *[round(float(r), 4) for r in shaped_rewards],
                *[round(float(e), 4) for e in epsilons],
                next_state,
                round(total_reward, 4),
                coop_flag,
                ma_coop if ma_coop == "" else round(float(ma_coop), 4),
                ma_reward if ma_reward == "" else round(float(ma_reward), 4),
                extra.get("total_extraction", ""),
                extra.get("stock_level", ""),
            ])

            state = next_state

    return csv_path, time_to_threshold


def plot_metric(per_run_paths: Dict[float, str], log_dir: str, tag: str, metric_field: str, ylabel: str, out_name: str):
    plt.figure()
    for emp, path in per_run_paths.items():
        xs, ys = [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row[metric_field] == "":
                    continue
                xs.append(int(row["step"]))
                ys.append(float(row[metric_field]))
        if xs:
            plt.plot(xs, ys, label=f"α={emp}")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(f"{tag}: {ylabel} vs steps (N={NUM_AGENTS})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, out_name), dpi=300)


def main():
    random.seed(SEED)

    for game_name in RUN_GAMES:
        env, _, log_dir, _, tag = build_game(game_name)
        ensure_dir(log_dir)

        summary_path = os.path.join(log_dir, f"summary_{tag}_N{NUM_AGENTS}.csv")
        summary_rows = []
        per_paths: Dict[float, str] = {}

        for emp in EMPATHY_ALPHAS:
            csv_path, ttc = run_experiment(game_name, emp)
            per_paths[emp] = csv_path
            summary_rows.append([tag, emp, EPISODES, ttc])

        # write summary
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["game", "empathy_alpha", "steps", "time_to_coop_threshold"])
            w.writerows(summary_rows)

        print(f"\nDone: {tag}")
        print(f"Logs: {os.path.abspath(log_dir)}")
        print(f"Summary: {os.path.abspath(summary_path)}")

        # plots
        plot_metric(
            per_paths,
            log_dir,
            tag,
            metric_field="moving_avg_coop",
            ylabel="Moving avg cooperation",
            out_name=f"coop_trends_{tag}_N{NUM_AGENTS}.png",
        )
        plot_metric(
            per_paths,
            log_dir,
            tag,
            metric_field="moving_avg_total_reward",
            ylabel="Moving avg total reward",
            out_name=f"total_reward_trends_{tag}_N{NUM_AGENTS}.png",
        )


if __name__ == "__main__":
    main()