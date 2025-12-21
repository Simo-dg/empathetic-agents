# ===============================
# EXPERIMENT TYPE: STAG HUNT
# ===============================

# Actions
ACTIONS = ["S", "H"]   # S = Stag, H = Hare

# 2-player Stag Hunt payoff matrix
# payoff[(a_i, a_j)] = (r_i, r_j)
PAYOFF_MATRIX = {
    ("S", "S"): (4, 4),
    ("S", "H"): (0, 3),
    ("H", "S"): (3, 0),
    ("H", "H"): (2, 2),
}

# ===============================
# MULTI-AGENT SETTINGS
# ===============================

NUM_AGENTS = 4

# ===============================
# Q-LEARNING PARAMETERS
# ===============================

ALPHA = 0.2
GAMMA = 0.95

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# ===============================
# TRAINING
# ===============================

EPISODES = 5000
EVAL_WINDOW = 100

# ===============================
# EMPATHY EXPERIMENT
# ===============================

# empathy_alpha = weight on own reward
# 1.0 = selfish
EMPATHY_ALPHAS = [1.0, 0.8, 0.5, 0.2]

COOP_THRESHOLD = 0.8

# ===============================
# LOGGING / REPRODUCIBILITY
# ===============================

SEED = 42
LOG_DIR = "logs_stag"