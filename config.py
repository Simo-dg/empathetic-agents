# Global settings for quick tweaks

# Payoff matrix for Prisoner's Dilemma (R, S, T, P)
# Rewards: (A, B)
PD_PAYOFFS = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

ACTIONS = ["C", "D"]

# Q-learning params
ALPHA = 0.2      # learning rate
GAMMA = 0.95     # discount factor
EPS_START = 1.0  # initial epsilon
EPS_END = 0.05   # min epsilon
EPS_DECAY = 0.995

EPISODES = 5000
EVAL_WINDOW = 100  # for moving averages and summaries
SEED = 42

# Empathy levels to test (alpha = weight on own reward)
# alpha=1.0 -> selfish; lower alpha -> more empathic
ALPHAS = [1.0, 0.8, 0.5, 0.2]

# Logging
LOG_DIR = "logs"
COOP_THRESHOLD = 0.8  # for time-to-cooperation metric