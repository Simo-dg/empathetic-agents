# resource_environment.py

import random
from typing import Optional, Sequence, Tuple


class ResourceSharingEnv:
    """
    N-agent resource sharing with a common renewable stock.

    - Shared stock S.
    - Each agent chooses extraction in {0,1,2}.
    - If total extraction <= stock:
        reward_i = extraction_i
      else:
        reward_i = penalty (everyone)
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

    def step(self, actions: Sequence[int]) -> Tuple[Tuple[float, ...], int]:
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        for a in actions:
            if a not in (0, 1, 2):
                raise ValueError(f"Invalid action {a}. Must be 0, 1 or 2.")

        total_extraction = sum(actions)

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
        return tuple(rewards), next_state