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
# multi-seed evaluation (robustness)
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# --- New baselines ---
BASELINES = ["empathy", "team", "random", "hetero_fixed", "learn_alpha"]

# --- Heterogeneous empathy profiles (examples) ---
# If N is 4, you can use one of these vectors.
HETERO_PROFILES = [
    [1.0, 1.0, 1.0, 1.0],      # all selfish (same as empathy α=1)
    [0.8, 0.8, 0.2, 0.2],      # two more selfish, two more prosocial
    [1.0, 0.5, 0.5, 0.0],      # diverse mix
]

# --- Learning empathy (bandit over alpha) ---
ALPHA_CANDIDATES = [0.0, 0.2, 0.5, 0.8, 1.0]
ALPHA_BANDIT_EPS = 0.1          # exploration for choosing alpha
LEARN_ALPHA_SIGNAL = "team"     # "self" | "team" | "shaped"

# empathy sweep
EMPATHY_ALPHAS = [1.0, 0.8, 0.5, 0.2, 0.0]

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