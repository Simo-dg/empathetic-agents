import random
from typing import Tuple, Optional

class MatrixGameEnv:
    """
    Stateless 2-player matrix game (e.g., Prisoner's Dilemma).
    Observation = previous joint action (for a simple Markov signal), or None at start.
    """
    def __init__(self, payoff_table, actions, seed: Optional[int] = None):
        self.payoff = payoff_table
        self.actions = actions
        self.rng = random.Random(seed)
        self.prev_joint_action = None  # e.g., ("C","D") from last step

    def reset(self):
        self.prev_joint_action = None
        # observation shown to both agents; identical state for both
        return self.prev_joint_action

    def step(self, a1: str, a2: str) -> Tuple[Tuple[int, int], Tuple[str, str]]:
        assert a1 in self.actions and a2 in self.actions
        r1, r2 = self.payoff[(a1, a2)]
        self.prev_joint_action = (a1, a2)
        obs = self.prev_joint_action
        return (r1, r2), obs