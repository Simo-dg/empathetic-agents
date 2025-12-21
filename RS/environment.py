import random
from typing import Optional, Sequence, Tuple, Dict, Any


class MultiAgentMatrixGameEnv:
    """
    Stateless N-player matrix game built from pairwise interactions.
    Provide ANY 2-player payoff table:
        payoff[(a_i, a_j)] = (r_i, r_j)

    At each step:
      - all agents choose an action
      - each unordered pair (i, j) plays the 2-player game using payoff table
      - rewards are summed for each agent and divided by (N - 1)
        (so reward scale stays comparable to 2-player case)

    Observation/state = previous joint action profile (tuple of actions), or None at start.
    """

    def __init__(
        self,
        payoff_table: Dict[Tuple[Any, Any], Tuple[float, float]],
        actions: Sequence[Any],
        num_agents: int = 2,
        seed: Optional[int] = None,
    ):
        assert num_agents >= 2, "Need at least 2 agents"
        self.payoff = payoff_table
        self.actions = list(actions)
        self.num_agents = int(num_agents)
        self.rng = random.Random(seed)
        self.prev_joint_action: Optional[Tuple[Any, ...]] = None

    def reset(self):
        self.prev_joint_action = None
        return self.prev_joint_action

    def step(self, actions: Sequence[Any]) -> Tuple[Tuple[float, ...], Any, Dict[str, Any]]:
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        for a in actions:
            if a not in self.actions:
                raise ValueError(f"Invalid action: {a}")

        rewards = [0.0 for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                key = (actions[i], actions[j])
                r_i, r_j = self.payoff[key]
                rewards[i] += float(r_i)
                rewards[j] += float(r_j)

        scale = float(self.num_agents - 1)
        rewards = [r / scale for r in rewards]

        self.prev_joint_action = tuple(actions)
        obs = self.prev_joint_action
        extra = {}  # keep signature consistent with Resource env
        return tuple(rewards), obs, extra


class ResourceSharingEnv:
    """
    N-agent resource sharing environment with a common renewable stock.

    - Shared stock S.
    - Each agent chooses extraction: 0, 1, 2.
    - If total extraction <= stock: reward_i = extraction_i
    - If total extraction > stock: everyone gets penalty
    - Stock updates: S <- clamp(S - total_extraction + regen, 0, max_stock)

    State returned = discretized stock level (int, rounded).
    """

    def __init__(
        self,
        num_agents: int,
        max_stock: int,
        regen: float,
        penalty: float,
        seed: Optional[int] = None,
    ):
        assert num_agents >= 2
        self.num_agents = int(num_agents)
        self.max_stock = float(max_stock)
        self.regen = float(regen)
        self.penalty = float(penalty)

        self.rng = random.Random(seed)
        self.stock: float = self.max_stock

    def _get_state(self) -> int:
        return int(round(self.stock))

    def reset(self) -> int:
        self.stock = self.max_stock
        return self._get_state()

    def step(self, actions: Sequence[int]) -> Tuple[Tuple[float, ...], int, Dict[str, Any]]:
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        for a in actions:
            if a not in (0, 1, 2):
                raise ValueError(f"Invalid action {a}. Must be 0, 1, or 2.")

        total_extraction = int(sum(actions))

        if total_extraction <= self.stock:
            rewards = [float(a) for a in actions]
        else:
            rewards = [self.penalty for _ in range(self.num_agents)]

        self.stock = self.stock - total_extraction + self.regen
        if self.stock < 0.0:
            self.stock = 0.0
        if self.stock > self.max_stock:
            self.stock = self.max_stock

        next_state = self._get_state()
        extra = {"total_extraction": total_extraction, "stock_level": next_state}
        return tuple(rewards), next_state, extra