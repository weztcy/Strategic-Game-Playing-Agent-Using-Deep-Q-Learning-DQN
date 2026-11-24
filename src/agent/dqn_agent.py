"""
Deep Q-Network (DQN) Agent Implementation.
Supports Double DQN, target network synchronization, epsilon-greedy exploration,
checkpointing, and gradient clipping.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent.model import DuelingQNetwork, QNetwork
from src.agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent encapsulating policy network, target network, replay buffer,
    and action selection logic.
    """

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        grad_clip: float = 1.0,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build networks
        if use_dueling:
            self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        else:
            self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.memory = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.total_steps = 0
        self.training_updates = 0

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Selects an action using epsilon-greedy exploration.

        Returns:
            action: int (0, 1, or 2)
            q_values: np.ndarray of predicted Q-values (shape (3,))
        """
        curr_eps = self.epsilon if epsilon is None else epsilon

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values_tensor = self.policy_net(s_tensor)
            q_values = q_values_tensor.squeeze(0).cpu().numpy()

        if explore and self.rng.random() < curr_eps:
            action = int(self.rng.integers(0, self.action_dim))
        else:
            action = int(np.argmax(q_values))

        return action, q_values

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores experience in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Optional[float]:
        """
        Executes a single optimization step on a mini-batch of transitions.

        Returns:
            loss: float or None (if buffer has fewer samples than batch_size)
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, device=self.device
        )

        # Current Q-values for chosen actions: Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: action selected by policy_net, evaluated by target_net
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            # Bellman equation: y = r + (1 - done) * gamma * Q_target(s', a')
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Calculate loss
        loss = self.loss_fn(curr_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.training_updates += 1

        # Synchronize target network periodically
        if self.training_updates % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copies policy network weights to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Decays exploration parameter epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Saves agent checkpoint containing weights and optimizer state."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "training_updates": self.training_updates,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict:
        """Loads agent checkpoint from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.training_updates = checkpoint.get("training_updates", 0)
        return checkpoint.get("metadata", {})
