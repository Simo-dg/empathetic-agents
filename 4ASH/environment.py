import random
from typing import Optional, Sequence, Tuple


class MultiAgentMatrixGameEnv:
    """
    Stateless N-player matrix game built from pairwise Prisoner's Dilemma
    interactions between every pair of agents.

    - actions: list of possible actions, e.g. ["C", "D"]
    - payoff_table: dict mapping (a_i, a_j) -> (r_i, r_j) for the 2-player PD
    - num_agents: how many agents (>= 2)

    At each step:
      - all agents choose an action
      - each unordered pair (i, j) plays a PD using PD_PAYOFFS
      - rewards are summed for each agent and then divided by (num_agents - 1)
        so the scale stays comparable to the 2-player case

    Observation = previous joint action profile (tuple of actions), or None at the start.
    """

    def __init__(
        self,
        payoff_table,
        actions: Sequence[str],
        num_agents: int = 2,
        seed: Optional[int] = None,
    ):
        assert num_agents >= 2, "Need at least 2 agents"
        self.payoff = payoff_table
        self.actions = list(actions)
        self.num_agents = num_agents
        self.rng = random.Random(seed)
        self.prev_joint_action: Optional[Tuple[str, ...]] = None

    def reset(self):
        self.prev_joint_action = None
        return self.prev_joint_action

    def step(self, actions: Sequence[str]) -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
        assert len(actions) == self.num_agents, (
            f"Expected {self.num_agents} actions, got {len(actions)}"
        )
        for a in actions:
            assert a in self.actions, f"Invalid action: {a}"

        # initialise rewards
        rewards = [0.0 for _ in range(self.num_agents)]

        # pairwise Prisoner's Dilemma for each unordered pair (i, j)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                key = (actions[i], actions[j])
                r_i, r_j = self.payoff[key]
                rewards[i] += r_i
                rewards[j] += r_j

        # normalise so each agent's reward is comparable to 2-player case
        scale = float(self.num_agents - 1)
        rewards = [r / scale for r in rewards]

        self.prev_joint_action = tuple(actions)
        obs = self.prev_joint_action
        return tuple(rewards), obs