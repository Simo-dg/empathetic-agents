# ======================================
# GLOBAL TRAINING SETTINGS
# ======================================
LR = 0.2
GAMMA = 0.95

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

STEPS = 5000          # "episodes" = steps (these envs are continuing)
EVAL_WINDOW = 100
SEED = 42

# empathy sweep
EMPATHY_ALPHAS = [1.0, 0.8, 0.5, 0.2]

# "time to cooperation" threshold on moving average
COOP_THRESHOLD = 0.8

# ======================================
# PD (Prisoner's Dilemma)
# ======================================
PD_ACTIONS = ["C", "D"]
PD_PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# ======================================
# SH (Stag Hunt)
# ======================================
SH_ACTIONS = ["S", "H"]
SH_PAYOFFS = {
    ("S", "S"): (4, 4),
    ("S", "H"): (0, 3),
    ("H", "S"): (3, 0),
    ("H", "H"): (2, 2),
}

# ======================================
# RS (Resource Sharing)
# ======================================
RS_ACTIONS = [0, 1, 2]
RS_MAX_STOCK = 10
RS_REGEN = 2.0
RS_PENALTY = -0.5

# cooperation metric for RS:
RS_SUSTAINABLE_THRESHOLD = 4

# ======================================
# LOGGING
# ======================================
LOG_ROOT = "logs_all"

# ======================================
# EXPERIMENTS (THE “4 MODELS”)
# ======================================
# Each dict defines one model.
# - type: "matrix" or "resource"
# - name: experiment label used in paths
# - N: number of agents
# - actions/payoff for matrix games
EXPERIMENTS = [
    # 1) PD, N=2
    {
        "name": "pd_N2",
        "type": "matrix",
        "N": 2,
        "actions": PD_ACTIONS,
        "payoff": PD_PAYOFFS,
        "coop_action": "C",  # coop if all agents choose this
    },
    # 2) PD, N=4
    {
        "name": "pd_N4",
        "type": "matrix",
        "N": 4,
        "actions": PD_ACTIONS,
        "payoff": PD_PAYOFFS,
        "coop_action": "C",
    },
    # 3) SH, N=4
    {
        "name": "sh_N4",
        "type": "matrix",
        "N": 4,
        "actions": SH_ACTIONS,
        "payoff": SH_PAYOFFS,
        "coop_action": "S",
    },
    # 4) RS, N=4
    {
        "name": "rs_N4",
        "type": "resource",
        "N": 4,
        "actions": RS_ACTIONS,
        "max_stock": RS_MAX_STOCK,
        "regen": RS_REGEN,
        "penalty": RS_PENALTY,
        "sustainable_threshold": RS_SUSTAINABLE_THRESHOLD,
    },
]