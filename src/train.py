# train.py
import os
import csv
import random
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import matplotlib.pyplot as plt

from agent import QAgent
from environment import MultiAgentMatrixGameEnv, ResourceSharingEnv
from config import (
    LR, GAMMA, EPS_START, EPS_END, EPS_DECAY,
    STEPS, EVAL_WINDOW, SEED,
    EMPATHY_ALPHAS, COOP_THRESHOLD,
    LOG_ROOT, EXPERIMENTS,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(vals: deque) -> float:
    return sum(vals) / len(vals)


def coop_flag_matrix(joint_actions: List[Any], coop_action: Any) -> int:
    return 1 if all(a == coop_action for a in joint_actions) else 0


def coop_flag_resource(extra: Dict[str, Any], sustainable_threshold: int) -> int:
    return 1 if int(extra["total_extraction"]) <= int(sustainable_threshold) else 0


def run_one_experiment(exp: Dict[str, Any], empathy_alpha: float) -> Tuple[str, Optional[int]]:
    name = exp["name"]
    exp_dir = os.path.join(LOG_ROOT, name)
    ensure_dir(exp_dir)

    N = int(exp["N"])
    actions_space = exp["actions"]

    # build env
    if exp["type"] == "matrix":
        env = MultiAgentMatrixGameEnv(
            payoff_table=exp["payoff"],
            actions=actions_space,
            num_agents=N,
            seed=SEED,
        )
        coop_action = exp["coop_action"]
        coop_fn = lambda joint_actions, _state, _extra: coop_flag_matrix(joint_actions, coop_action)

    elif exp["type"] == "resource":
        env = ResourceSharingEnv(
            num_agents=N,
            max_stock=exp["max_stock"],
            regen=exp["regen"],
            penalty=exp["penalty"],
            seed=SEED,
        )
        sustainable_threshold = exp["sustainable_threshold"]
        coop_fn = lambda _joint_actions, _state, extra: coop_flag_resource(extra, sustainable_threshold)

    else:
        raise ValueError(f"Unknown experiment type: {exp['type']}")

    # agents
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
        for i in range(N)
    ]

    csv_path = os.path.join(exp_dir, f"per_step_empathy_{empathy_alpha:.2f}.csv")

    fieldnames = (
        ["step", "experiment", "type", "N", "empathy_alpha"]
        + [f"A{i+1}_action" for i in range(N)]
        + [f"r{i+1}" for i in range(N)]
        + [f"r{i+1}_shaped" for i in range(N)]
        + [f"epsilon_A{i+1}" for i in range(N)]
        + ["state", "total_reward", "coop_flag", "moving_avg_coop", "moving_avg_total_reward"]
        + ["extra_total_extraction", "extra_stock_level"]
    )

    coop_window = deque(maxlen=EVAL_WINDOW)
    reward_window = deque(maxlen=EVAL_WINDOW)
    time_to_threshold = None

    state = env.reset()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)

        for step in range(1, STEPS + 1):
            joint_actions = [ag.act(state) for ag in agents]
            rewards, next_state, extra = env.step(joint_actions)

            total_reward = float(sum(rewards))

            shaped_rewards = []
            epsilons = []
            for i, ag in enumerate(agents):
                r_self = float(rewards[i])
                r_others_avg = (total_reward - r_self) / (N - 1)

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

            w.writerow([
                step, name, exp["type"], N, empathy_alpha,
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


def plot_metric(per_paths: Dict[float, str], out_dir: str, title: str, field: str, ylabel: str, out_name: str):
    plt.figure()
    for emp, path in per_paths.items():
        xs, ys = [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row[field] == "":
                    continue
                xs.append(int(row["step"]))
                ys.append(float(row[field]))
        if xs:
            plt.plot(xs, ys, label=f"α={emp}")

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, out_name), dpi=300)


def main():
    random.seed(SEED)
    ensure_dir(LOG_ROOT)

    # global summary over ALL experiments
    summary_path = os.path.join(LOG_ROOT, "summary_all.csv")
    summary_rows = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        exp_dir = os.path.join(LOG_ROOT, name)
        ensure_dir(exp_dir)

        per_paths: Dict[float, str] = {}

        for emp in EMPATHY_ALPHAS:
            csv_path, ttc = run_one_experiment(exp, emp)
            per_paths[emp] = csv_path
            summary_rows.append([name, exp["type"], exp["N"], emp, STEPS, ttc])

        # plots per experiment
        plot_metric(
            per_paths,
            exp_dir,
            title=f"{name}: moving avg cooperation (N={exp['N']})",
            field="moving_avg_coop",
            ylabel="Moving avg cooperation",
            out_name="coop_trends.png",
        )
        plot_metric(
            per_paths,
            exp_dir,
            title=f"{name}: moving avg total reward (N={exp['N']})",
            field="moving_avg_total_reward",
            ylabel="Moving avg total reward",
            out_name="total_reward_trends.png",
        )

        print(f"Done: {name}  -> {os.path.abspath(exp_dir)}")

    # write combined summary
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "type", "N", "empathy_alpha", "steps", "time_to_coop_threshold"])
        w.writerows(summary_rows)

    print(f"\nALL DONE. Summary: {os.path.abspath(summary_path)}")
    print(f"Logs root: {os.path.abspath(LOG_ROOT)}")


if __name__ == "__main__":
    main()