"""
Neural Network Architectures for Deep Q-Learning.
Includes standard Multi-Layer Perceptron DQN and Dueling DQN.
"""

from typing import List, Tuple
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Standard Deep Q-Network MLP.
    Maps an observation state vector to action-values Q(s, a).
    """

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        Returns:
            Tensor of shape (batch_size, action_dim) containing Q-values
        """
        return self.net(state)


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network.
    Decomposes Q(s, a) into Value V(s) and Advantage A(s, a):
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    """

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_vals
