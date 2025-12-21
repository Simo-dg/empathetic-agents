import random
from collections import defaultdict
from typing import Dict, Hashable, List, Tuple

State = Hashable  # we only require that states are hashable (None, tuples, etc.)


class QAgent:
    def __init__(
        self,
        name: str,
        actions: List[str],
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        seed: int = 0,
        empathy_alpha: float = 1.0,
    ):
        """
        empathy_alpha ∈ [0, 1] = weight on own reward.
        1.0 = purely selfish.

        Effective reward used to update Q:
            r_eff = empathy_alpha * r_self + (1 - empathy_alpha) * r_others_avg

        where r_others_avg is the average reward of all *other* agents.
        """
        self.name = name
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = eps_start
        self.empathy_alpha = empathy_alpha

        self.rng = random.Random(seed)
        self.Q: Dict[Tuple[State, str], float] = defaultdict(float)

    def _epsilon_greedy_action(self, state: State) -> str:
        # exploration
        if self.rng.random() < self.eps:
            return self.rng.choice(self.actions)

        # exploitation: choose argmax_a Q(s, a)
        qvals = [(self.Q[(state, a)], a) for a in self.actions]
        qvals.sort(key=lambda x: x[0], reverse=True)
        return qvals[0][1]

    def act(self, state: State) -> str:
        """Pick an action given the current state."""
        return self._epsilon_greedy_action(state)

    def update(
        self,
        state: State,
        action: str,
        reward_self: float,
        reward_others_avg: float,
        next_state: State,
    ) -> None:
        """Standard Q-learning update with empathy-shaped reward."""
        shaped = (
            self.empathy_alpha * reward_self
            + (1 - self.empathy_alpha) * reward_others_avg
        )
        max_next = max(self.Q[(next_state, a)] for a in self.actions)
        td_target = shaped + self.gamma * max_next
        td = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)