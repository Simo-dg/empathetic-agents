# ===============================
# RESOURCE SHARING CONFIG (RS)
# ===============================

# Agents + action space
NUM_AGENTS = 4
ACTIONS = [0, 1, 2]  # extraction levels

# Environment dynamics
MAX_STOCK = 10
REGEN = 2.0
PENALTY = -0.5
SEED = 42

# Q-learning
LR = 0.2           # learning rate (Q alpha)
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# Training
EPISODES = 5000
EVAL_WINDOW = 100

# Empathy sweep (weight on own reward)
EMPATHY_ALPHAS = [1.0, 0.8, 0.5, 0.2]

# Logging
LOG_DIR = "logs_resource"

# "Cooperation" metric for RS:
# cooperative if total extraction <= threshold
SUSTAINABLE_THRESHOLD = 4
COOP_THRESHOLD = 0.8