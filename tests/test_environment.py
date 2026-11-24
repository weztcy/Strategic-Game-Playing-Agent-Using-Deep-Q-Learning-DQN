"""
Unit tests for Snake Environment.
Validates state shapes, actions, collision detection, food eating, and rewards.
"""

import math
import numpy as np
import pytest

from src.environment.game import Action, Direction, Point, SnakeGame


def test_environment_initialization():
    env = SnakeGame(grid_width=20, grid_height=20, seed=42)
    state, info = env.reset(seed=42)

    assert state.shape == (11,)
    assert state.dtype == np.float32
    assert np.all((state == 0.0) | (state == 1.0))
    assert info["score"] == 0
    assert info["snake_length"] == 3
    assert len(env.snake) == 3
    assert env.direction == Direction.RIGHT


def test_environment_step_validity():
    env = SnakeGame(grid_width=20, grid_height=20, seed=42)
    state, _ = env.reset(seed=42)

    for action in [Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_LEFT]:
        next_state, reward, terminated, truncated, info = env.step(action)
        assert next_state.shape == (11,)
        assert next_state.dtype == np.float32
        assert not math.isnan(reward)
        assert not math.isinf(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "score" in info


def test_invalid_action_rejection():
    env = SnakeGame(grid_width=20, grid_height=20, seed=42)
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_food_consumption_and_growth():
    env = SnakeGame(grid_width=20, grid_height=20, seed=42)
    env.reset(seed=42)

    # Place food directly in front of the snake's head
    head = env.snake[0]
    env.food = Point(head.x + 1, head.y)
    prev_len = len(env.snake)
    prev_score = env.score

    obs, reward, term, trunc, info = env.step(Action.STRAIGHT)

    assert env.score == prev_score + 1
    assert len(env.snake) == prev_len + 1
    assert reward >= env.food_reward - 0.2  # Food reward included


def test_wall_collision():
    env = SnakeGame(grid_width=10, grid_height=10, seed=42)
    env.reset(seed=42)

    # Place snake near right edge facing right
    env.direction = Direction.RIGHT
    env.snake = [Point(9, 5), Point(8, 5), Point(7, 5)]

    # Moving straight hits right boundary
    obs, reward, terminated, truncated, info = env.step(Action.STRAIGHT)
    assert terminated is True
    assert reward == env.collision_penalty
    assert info["reason"] == "collision"


def test_self_collision():
    env = SnakeGame(grid_width=10, grid_height=10, seed=42)
    env.reset(seed=42)

    # Construct snake in a loop about to collide with its body
    env.direction = Direction.UP
    env.snake = [
        Point(5, 5),  # Head
        Point(5, 6),
        Point(6, 6),
        Point(6, 5),  # Body to the right of head
    ]

    # Turn right moves from (5,5) facing UP to (6,5) which hits its body
    obs, reward, terminated, truncated, info = env.step(Action.TURN_RIGHT)
    assert terminated is True
    assert reward == env.collision_penalty
    assert info["reason"] == "collision"


def test_starvation_timeout():
    env = SnakeGame(grid_width=10, grid_height=10, max_idle_steps_multiplier=1, seed=42)
    env.reset(seed=42)

    # Manually trigger max idle steps limit
    max_idle = max(100, len(env.snake) * 1)
    env.idle_steps = max_idle - 1

    obs, reward, terminated, truncated, info = env.step(Action.STRAIGHT)
    assert truncated is True
    assert info["reason"] == "starvation_timeout"
