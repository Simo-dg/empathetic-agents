import random
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Tuple, Optional

State = Hashable


class QAgent:
    """
    Tabular Q-learning agent with "empathy reward shaping":
        shaped = alpha * r_self + (1-alpha) * r_others_avg

    Extensions:
      - heterogeneous fixed alpha per agent (just pass empathy_alpha)
      - learning alpha via a simple bandit over discrete alpha candidates
    """

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
        empathy_mode: str = "fixed",  # "fixed" | "bandit"
        alpha_candidates: Optional[List[float]] = None,
        alpha_bandit_eps: float = 0.1,
    ):
        self.name = name
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)

        self.rng = random.Random(seed)

        # empathy / alpha control
        self.empathy_mode = empathy_mode
        self.empathy_alpha = float(empathy_alpha)

        self.alpha_candidates = list(alpha_candidates) if alpha_candidates else [float(empathy_alpha)]
        self.alpha_bandit_eps = float(alpha_bandit_eps)

        # bandit state for alpha selection
        self.alpha_value: Dict[float, float] = {a: 0.0 for a in self.alpha_candidates}
        self.alpha_count: Dict[float, int] = {a: 0 for a in self.alpha_candidates}

        # current alpha used this step
        self.current_alpha: float = float(empathy_alpha)

        # Q-table
        self.Q: Dict[Tuple[State, Any], float] = defaultdict(float)

    # -------- action selection --------
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

    # -------- empathy alpha selection --------
    def select_empathy_alpha(self) -> float:
        """
        Sets self.current_alpha.
        - fixed mode: uses self.empathy_alpha
        - bandit mode: epsilon-greedy over alpha_candidates based on alpha_value
        """
        if self.empathy_mode != "bandit":
            self.current_alpha = float(self.empathy_alpha)
            return self.current_alpha

        # epsilon-greedy over alpha candidates
        if self.rng.random() < self.alpha_bandit_eps:
            a = self.rng.choice(self.alpha_candidates)
        else:
            # tie-break randomly among best
            best_val = max(self.alpha_value[a] for a in self.alpha_candidates)
            best = [a for a in self.alpha_candidates if self.alpha_value[a] == best_val]
            a = self.rng.choice(best)

        self.current_alpha = float(a)
        return self.current_alpha

    def update_alpha_bandit(self, signal: float) -> None:
        """
        Incremental mean update for the selected alpha arm.
        """
        if self.empathy_mode != "bandit":
            return
        a = float(self.current_alpha)
        self.alpha_count[a] += 1
        n = self.alpha_count[a]
        self.alpha_value[a] += (float(signal) - self.alpha_value[a]) / float(n)

    # -------- learning update --------
    def update(
        self,
        state: State,
        action: Any,
        reward_self: float,
        reward_others_avg: float,
        next_state: State,
    ) -> None:
        # Use the alpha selected for this step (or fixed alpha)
        alpha = float(self.current_alpha) if self.empathy_mode == "bandit" else float(self.empathy_alpha)

        shaped = alpha * float(reward_self) + (1.0 - alpha) * float(reward_others_avg)

        max_next = max(self.Q[(next_state, a)] for a in self.actions)
        td_target = shaped + self.gamma * max_next
        td = td_target - self.Q[(state, action)]
        self.Q[(state, action)] += self.alpha * td

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)