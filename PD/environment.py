import random
from typing import Optional, Sequence, Tuple, Dict, List

PayoffTable = Dict[Tuple[str, str], Tuple[float, float]]


class MultiAgentMatrixGameEnv:
    """
    Unified stateless N-agent matrix game built from pairwise interactions.

    - payoff[(a_i, a_j)] -> (r_i, r_j)
    - rewards summed over all unordered pairs (i, j)
    - normalized by (num_agents - 1) so scale matches 2-agent case
    - observation/state: previous joint action profile (tuple of actions), or None at reset
    """

    def __init__(
        self,
        payoff_table: PayoffTable,
        actions: Sequence[str],
        num_agents: int,
        seed: Optional[int] = None,
    ):
        if num_agents < 2:
            raise ValueError("num_agents must be >= 2")

        self.payoff = payoff_table
        self.actions = list(actions)
        self.num_agents = num_agents
        self.rng = random.Random(seed)
        self.prev_joint_action: Optional[Tuple[str, ...]] = None

    def reset(self) -> Optional[Tuple[str, ...]]:
        self.prev_joint_action = None
        return self.prev_joint_action

    def step(self, actions: Sequence[str]) -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        for a in actions:
            if a not in self.actions:
                raise ValueError(f"Invalid action: {a}")

        rewards: List[float] = [0.0] * self.num_agents

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                r_i, r_j = self.payoff[(actions[i], actions[j])]
                rewards[i] += float(r_i)
                rewards[j] += float(r_j)

        scale = float(self.num_agents - 1)
        rewards = [r / scale for r in rewards]

        self.prev_joint_action = tuple(actions)
        return tuple(rewards), self.prev_joint_action