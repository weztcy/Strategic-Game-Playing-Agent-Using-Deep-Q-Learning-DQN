"""
Integration tests for Training Loop and Evaluation Pipelines.
"""

import tempfile
import numpy as np
import pytest

from src.agent.dqn_agent import DQNAgent
from src.environment.game import SnakeGame
from src.evaluation.baselines import HeuristicAgent, RandomAgent
from src.evaluation.evaluate import evaluate_agent
from src.training.train import train_dqn


def test_training_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = f"{tmpdir}/models"
        log_dir = f"{tmpdir}/metrics"

        agent = train_dqn(
            episodes=10,
            seed=42,
            save_dir=save_dir,
            log_dir=log_dir,
            smoke_test=False,
        )

        assert agent.total_steps > 0
        assert (agent.training_updates > 0) or (len(agent.memory) < agent.batch_size)


def test_evaluation_pipeline():
    config = {
        "environment": {"grid_width": 10, "grid_height": 10},
        "rewards": {
            "food_reward": 10.0,
            "collision_penalty": -10.0,
            "closer_reward": 0.1,
            "farther_penalty": -0.15,
            "step_penalty": -0.01,
        },
    }

    random_agent = RandomAgent(action_dim=3, seed=42)
    heuristic_agent = HeuristicAgent(seed=42)
    dqn_agent = DQNAgent(state_dim=11, action_dim=3, epsilon_start=0.0)

    for ag, name in [
        (random_agent, "Random"),
        (heuristic_agent, "Heuristic"),
        (dqn_agent, "DQN"),
    ]:
        df, summary = evaluate_agent(
            agent=ag,
            agent_name=name,
            episodes=5,
            seeds=[42, 101],
            config=config,
            success_threshold=5,
        )

        assert len(df) == 5
        assert summary["avg_score"] >= 0
        assert "success_rate" in summary
        assert not np.isnan(summary["avg_reward"])
