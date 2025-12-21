# =========================
# Unified experiment config
# Runs BOTH: first N=2, then N=4
# =========================

NUM_AGENTS_LIST = [2, 4]

# Environment
ACTIONS = ["C", "D"]

PD_PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# Q-learning
ALPHA = 0.2
GAMMA = 0.95

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# Training
EPISODES = 5000
EVAL_WINDOW = 100
SEED = 42

# Empathy (weight on own reward)
ALPHAS = [1.0, 0.8, 0.5, 0.2]

# Logging / evaluation
LOG_DIR = "logs"
COOP_THRESHOLD = 0.8