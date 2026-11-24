"""
Experience Replay Buffer for Deep Q-Learning.
"""

from collections import deque
import random
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity cyclic buffer storing experience transitions (s, a, r, s', done).
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores a transition in the buffer."""
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly samples a batch of transitions and returns them as PyTorch tensors.
        """
        transitions = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        b_states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        b_actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        b_rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        b_next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        b_dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return b_states, b_actions, b_rewards, b_next_states, b_dones

    def __len__(self) -> int:
        return len(self.buffer)
