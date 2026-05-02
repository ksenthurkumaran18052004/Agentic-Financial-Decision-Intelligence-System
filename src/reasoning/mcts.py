"""
Monte Carlo Tree Search for fraud decision reasoning.

Actions: APPROVE, FLAG, BLOCK
Each simulation perturbs the evidence slightly (models uncertainty) and computes
a reward for each action. UCB1 selects which action to explore next.
After N simulations the action with the highest mean reward wins.

Reward function:
  APPROVE: high reward when risk is low, heavy penalty when risk is high
  FLAG:    moderate flat reward (always safe to ask for review)
  BLOCK:   high reward when risk is high, penalty when blocking legitimate tx
"""

import math
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MCTS_SIMULATIONS, MCTS_EXPLORATION

ACTIONS = ["APPROVE", "FLAG", "BLOCK"]


def _reward(action: str, risk: float, anomaly: float, rag_hits: int) -> float:
    """Compute reward for taking `action` given evidence."""
    fraud_signal = (
        0.5 * risk
        + 0.3 * max(0.0, -anomaly)        # anomaly is negative when anomalous
        + 0.2 * min(1.0, rag_hits / 4.0)  # more relevant rules = stronger signal
    )
    fraud_signal = min(1.0, max(0.0, fraud_signal))

    if action == "APPROVE":
        return 1.0 - fraud_signal          # good when legitimate, bad when fraud
    elif action == "FLAG":
        return 0.45 + 0.1 * fraud_signal   # always decent; slightly better when risky
    else:  # BLOCK
        return fraud_signal                 # good when fraudulent, bad when legitimate


class _Node:
    def __init__(self, action: str):
        self.action = action
        self.visits = 0
        self.total_reward = 0.0

    def ucb1(self, total_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        exploration  = c * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration

    def update(self, reward: float):
        self.visits += 1
        self.total_reward += reward

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0


class MCTSReasoner:
    def __init__(self, risk_score: float, anomaly_score: float, rag_hits: int = 0):
        self.risk_score    = risk_score
        self.anomaly_score = anomaly_score
        self.rag_hits      = rag_hits
        self.nodes         = {a: _Node(a) for a in ACTIONS}

    def _simulate(self, action: str) -> float:
        """One simulation: perturb evidence, compute reward."""
        noise = random.gauss(0, 0.05)
        risk    = min(1.0, max(0.0, self.risk_score    + noise))
        anomaly = min(0.5, max(-1.0, self.anomaly_score + noise * 0.5))
        return _reward(action, risk, anomaly, self.rag_hits)

    def search(self) -> tuple[str, int]:
        total = 0
        for _ in range(MCTS_SIMULATIONS):
            # UCB1 selection
            action = max(
                ACTIONS,
                key=lambda a: self.nodes[a].ucb1(max(1, total), MCTS_EXPLORATION),
            )
            reward = self._simulate(action)
            self.nodes[action].update(reward)
            total += 1

        best = max(ACTIONS, key=lambda a: self.nodes[a].mean_reward)
        return best, MCTS_SIMULATIONS

    def summary(self) -> dict:
        return {
            a: {"visits": n.visits, "mean_reward": round(n.mean_reward, 4)}
            for a, n in self.nodes.items()
        }
