import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

class QAgent:
    def __init__(self, name: str, actions: List[str], alpha: float, gamma: float,
                 eps_start: float, eps_end: float, eps_decay: float, seed: int = 0,
                 empathy_alpha: float = 1.0):
        """
        empathy_alpha (∈ [0,1]) = weight on own reward. 1.0 = selfish.
        Effective reward used to update Q: r' = empathy_alpha * r_self + (1 - empathy_alpha) * r_other
        """
        self.name = name
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = random.Random(seed)
        self.empathy_alpha = empathy_alpha

        # Q[(state, action)] -> value
        self.Q: Dict[Tuple[Any, str], float] = defaultdict(float)

    def select_action(self, state) -> str:
        if self.rng.random() < self.eps:
            return self.rng.choice(self.actions)
        # greedy
        qvals = [(self.Q[(state, a)], a) for a in self.actions]
        qvals.sort(reverse=True)
        return qvals[0][1]

    def update(self, state, action, reward_self: float, reward_other: float, next_state):
        # empathy-shaped reward
        shaped = self.empathy_alpha * reward_self + (1 - self.empathy_alpha) * reward_other
        # target
        max_next = max(self.Q[(next_state, a)] for a in self.actions)
        td_target = shaped + self.gamma * max_next
        td = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)