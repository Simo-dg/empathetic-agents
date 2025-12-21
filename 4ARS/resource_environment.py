# resource_environment.py

import random
from typing import Optional, Sequence, Tuple


class ResourceSharingEnv:
    """
    N-agent resource sharing environment with a common renewable stock.

    - There is a shared stock S (e.g. fish in a lake).
    - Each agent chooses how much to extract: 0 (low), 1 (medium), 2 (high).
    - If the total extraction <= current stock:
        - Each agent gets reward = amount they extracted.
    - If total extraction > stock (overuse / "tragedy"):
        - Everyone gets a small negative reward (penalty).
    - After each step, the stock regenerates a bit but is capped at max_stock.

    State returned to agents = discretised current stock level (int).
    """

    def __init__(
        self,
        num_agents: int = 4,
        max_stock: int = 10,
        regen: float = 2.0,
        penalty: float = -0.5,
        seed: Optional[int] = None,
    ):
        assert num_agents >= 2
        self.num_agents = num_agents
        self.max_stock = float(max_stock)
        self.regen = float(regen)
        self.penalty = float(penalty)

        self.rng = random.Random(seed)
        self.stock: float = self.max_stock

    def _get_state(self) -> int:
        # simple discrete state: current stock rounded to nearest int
        return int(round(self.stock))

    def reset(self) -> int:
        # start with full stock
        self.stock = self.max_stock
        return self._get_state()

    def step(self, actions: Sequence[int]) -> Tuple[Tuple[float, ...], int]:
        """
        actions: list/tuple of extraction decisions for each agent (0, 1, or 2).
        Returns:
          rewards: tuple of float rewards for each agent
          next_state: integer stock level
        """
        assert len(actions) == self.num_agents, (
            f"Expected {self.num_agents} actions, got {len(actions)}"
        )

        # make sure actions are in {0,1,2}
        for a in actions:
            if a not in (0, 1, 2):
                raise ValueError(f"Invalid action {a}. Must be 0, 1 or 2.")

        total_extraction = sum(actions)

        if total_extraction <= self.stock:
            # sustainable use: everyone gets what they took
            rewards = [float(a) for a in actions]
        else:
            # overuse: tragedy of the commons
            rewards = [self.penalty for _ in range(self.num_agents)]

        # stock dynamics: decrease by extraction, then regenerate
        self.stock = self.stock - total_extraction + self.regen
        # stock is bounded
        if self.stock < 0.0:
            self.stock = 0.0
        if self.stock > self.max_stock:
            self.stock = self.max_stock

        next_state = self._get_state()
        return tuple(rewards), next_state# -*- coding: utf-8 -*-

