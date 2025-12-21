import random
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Tuple

State = Hashable


class QAgent:
    def __init__(
        self,
        name: str,
        actions: List[Any],
        alpha: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        seed: int = 0,
        empathy_alpha: float = 1.0,
    ):
        self.name = name
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)

        self.empathy_alpha = float(empathy_alpha)
        self.rng = random.Random(seed)

        self.Q: Dict[Tuple[State, Any], float] = defaultdict(float)

    def act(self, state: State) -> Any:
        # epsilon-greedy + random tie-break
        if self.rng.random() < self.eps:
            return self.rng.choice(self.actions)

        max_q = max(self.Q[(state, a)] for a in self.actions)
        best = [a for a in self.actions if self.Q[(state, a)] == max_q]
        return self.rng.choice(best)

    # backward-compatible alias
    def select_action(self, state: State) -> Any:
        return self.act(state)

    def update(
        self,
        state: State,
        action: Any,
        reward_self: float,
        reward_others_avg: float,
        next_state: State,
    ) -> None:
        shaped = (
            self.empathy_alpha * float(reward_self)
            + (1.0 - self.empathy_alpha) * float(reward_others_avg)
        )
        max_next = max(self.Q[(next_state, a)] for a in self.actions)
        td_target = shaped + self.gamma * max_next
        td = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)