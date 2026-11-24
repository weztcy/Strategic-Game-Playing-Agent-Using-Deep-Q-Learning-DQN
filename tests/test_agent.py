"""
Unit tests for DQN Agent, ReplayBuffer, and QNetwork.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from src.agent.dqn_agent import DQNAgent
from src.agent.model import DuelingQNetwork, QNetwork
from src.agent.replay_buffer import ReplayBuffer


def test_q_network_dimensions():
    model = QNetwork(state_dim=11, action_dim=3, hidden_dims=(128, 128))
    batch_states = torch.randn(8, 11)
    out = model(batch_states)
    assert out.shape == (8, 3)

    single_state = torch.randn(11)
    out_single = model(single_state)
    assert out_single.shape == (3,)


def test_dueling_q_network_dimensions():
    model = DuelingQNetwork(state_dim=11, action_dim=3, hidden_dim=128)
    batch_states = torch.randn(16, 11)
    out = model(batch_states)
    assert out.shape == (16, 3)


def test_replay_buffer_push_sample():
    buf = ReplayBuffer(capacity=100, seed=42)
    assert len(buf) == 0

    state = np.zeros(11, dtype=np.float32)
    next_state = np.ones(11, dtype=np.float32)

    for i in range(50):
        buf.push(state, action=i % 3, reward=1.0, next_state=next_state, done=False)

    assert len(buf) == 50

    b_s, b_a, b_r, b_ns, b_d = buf.sample(batch_size=16)
    assert b_s.shape == (16, 11)
    assert b_a.shape == (16, 1)
    assert b_r.shape == (16, 1)
    assert b_ns.shape == (16, 11)
    assert b_d.shape == (16, 1)


def test_dqn_action_selection():
    agent = DQNAgent(state_dim=11, action_dim=3, epsilon_start=0.0)
    state = np.zeros(11, dtype=np.float32)

    action, q_vals = agent.select_action(state, explore=False)
    assert action in [0, 1, 2]
    assert q_vals.shape == (3,)

    # With full exploration
    agent.epsilon = 1.0
    action_exp, _ = agent.select_action(state, explore=True)
    assert action_exp in [0, 1, 2]


def test_target_network_sync():
    agent = DQNAgent(state_dim=11, action_dim=3)

    # Modify policy net weights
    with torch.no_grad():
        for param in agent.policy_net.parameters():
            param.add_(1.0)

    # Check they differ before update
    policy_weight = next(agent.policy_net.parameters()).clone()
    target_weight = next(agent.target_net.parameters()).clone()
    assert not torch.allclose(policy_weight, target_weight)

    # Sync
    agent.update_target_network()
    target_weight_after = next(agent.target_net.parameters()).clone()
    assert torch.allclose(policy_weight, target_weight_after)


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        agent = DQNAgent(state_dim=11, action_dim=3, epsilon_start=0.75)
        agent.total_steps = 1234
        agent.save_checkpoint(ckpt_path, metadata={"custom_info": "ok"})

        assert ckpt_path.exists()

        new_agent = DQNAgent(state_dim=11, action_dim=3)
        meta = new_agent.load_checkpoint(ckpt_path)

        assert meta["custom_info"] == "ok"
        assert new_agent.epsilon == 0.75
        assert new_agent.total_steps == 1234

        # Verify weights match
        p1 = next(agent.policy_net.parameters())
        p2 = next(new_agent.policy_net.parameters())
        assert torch.allclose(p1, p2)
