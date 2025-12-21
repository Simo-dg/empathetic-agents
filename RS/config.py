# ===============================
# SELECT WHAT TO RUN
# ===============================
# options: "stag", "resource"
RUN_GAMES = ["stag", "resource"]

# ===============================
# SHARED TRAINING SETTINGS
# ===============================
NUM_AGENTS = 4

# Q-learning
LR = 0.2
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

EPISODES = 5000
EVAL_WINDOW = 100

SEED = 42

# Empathy sweep (weight on own reward)
EMPATHY_ALPHAS = [1.0, 0.8, 0.5, 0.2]

# Threshold for "time to cooperation"
COOP_THRESHOLD = 0.8

# ===============================
# STAG HUNT (SH)
# ===============================
STAG_ACTIONS = ["S", "H"]
STAG_PAYOFFS = {
    ("S", "S"): (4, 4),
    ("S", "H"): (0, 3),
    ("H", "S"): (3, 0),
    ("H", "H"): (2, 2),
}
LOG_DIR_STAG = "logs_stag"

# cooperation in SH = all choose "S"
def coop_flag_stag(joint_actions, _state, _extra) -> int:
    return 1 if all(a == "S" for a in joint_actions) else 0


# ===============================
# RESOURCE SHARING (RS)
# ===============================
RS_ACTIONS = [0, 1, 2]  # extraction
RS_MAX_STOCK = 10
RS_REGEN = 2.0
RS_PENALTY = -0.5
LOG_DIR_RESOURCE = "logs_resource"

# cooperation in RS = sustainable extraction (you can change this rule)
RS_SUSTAINABLE_THRESHOLD = 4

def coop_flag_resource(joint_actions, state, extra) -> int:
    # extra has: {"total_extraction": int, "stock_level": int}
    return 1 if extra["total_extraction"] <= RS_SUSTAINABLE_THRESHOLD else 0