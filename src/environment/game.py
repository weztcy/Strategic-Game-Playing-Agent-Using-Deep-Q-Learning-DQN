"""
Snake Game Environment for Reinforcement Learning (DQN).
Implements a standardized Markov Decision Process (MDP) for Snake.
"""

from collections import deque
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Action(IntEnum):
    STRAIGHT = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2


class Point(NamedTuple):
    x: int
    y: int


# Direction delta mappings (dx, dy) where (0,0) is top-left
DIRECTION_DELTAS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class SnakeGame:
    """
    Snake Environment conforming to standard RL step/reset semantics.
    Supports headless execution, reproducible seeding, relative action space,
    and structured observation vectors.
    """

    def __init__(
        self,
        grid_width: int = 20,
        grid_height: int = 20,
        max_idle_steps_multiplier: int = 100,
        food_reward: float = 10.0,
        collision_penalty: float = -10.0,
        closer_reward: float = 0.1,
        farther_penalty: float = -0.15,
        step_penalty: float = -0.01,
        seed: Optional[int] = None,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_idle_steps_multiplier = max_idle_steps_multiplier

        self.food_reward = food_reward
        self.collision_penalty = collision_penalty
        self.closer_reward = closer_reward
        self.farther_penalty = farther_penalty
        self.step_penalty = step_penalty

        self.rng = np.random.default_rng(seed)

        # Game state variables
        self.snake: List[Point] = []
        self.direction = Direction.RIGHT
        self.food = Point(0, 0)
        self.score = 0
        self.steps = 0
        self.idle_steps = 0
        self.game_over = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to the initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.direction = Direction.RIGHT
        head_x = self.grid_width // 2
        head_y = self.grid_height // 2

        self.snake = [
            Point(head_x, head_y),
            Point(head_x - 1, head_y),
            Point(head_x - 2, head_y),
        ]

        self.score = 0
        self.steps = 0
        self.idle_steps = 0
        self.game_over = False

        self._spawn_food()

        obs = self.get_state()
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
            "idle_steps": self.idle_steps,
        }
        return obs, info

    def _spawn_food(self) -> None:
        """Spawns food at a random grid location not occupied by the snake."""
        empty_cells = [
            Point(x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if Point(x, y) not in self.snake
        ]

        if not empty_cells:
            # Snake filled entire board - win condition
            self.food = Point(-1, -1)
            return

        idx = self.rng.integers(0, len(empty_cells))
        self.food = empty_cells[idx]

    def _get_new_direction(self, action: int) -> Direction:
        """
        Calculates new direction from relative action:
        0: Straight
        1: Turn Right (+1 mod 4)
        2: Turn Left (-1 mod 4)
        """
        if action == Action.STRAIGHT:
            return self.direction
        elif action == Action.TURN_RIGHT:
            return Direction((self.direction + 1) % 4)
        elif action == Action.TURN_LEFT:
            return Direction((self.direction - 1) % 4)
        else:
            raise ValueError(f"Invalid action {action}. Must be 0 (Straight), 1 (Right), or 2 (Left).")

    def _is_collision(self, pt: Point) -> bool:
        """Checks if a point collides with walls or snake body."""
        # Wall collision
        if pt.x < 0 or pt.x >= self.grid_width or pt.y < 0 or pt.y >= self.grid_height:
            return True
        # Body collision (excluding the tail if the snake doesn't grow this turn is standard,
        # but here we check against snake[:-1] or full snake)
        if pt in self.snake:
            return True
        return False

    def _manhattan_distance(self, p1: Point, p2: Point) -> int:
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executes an action in the environment.

        Args:
            action: 0 (Straight), 1 (Turn Right), 2 (Turn Left)

        Returns:
            obs: np.ndarray (shape (11,), dtype float32)
            reward: float
            terminated: bool (collision or win)
            truncated: bool (timeout / starvation loop limit)
            info: dict with game metrics
        """
        if self.game_over:
            raise RuntimeError("Episode has ended. Please call reset() before step().")

        self.steps += 1
        self.idle_steps += 1

        head = self.snake[0]
        prev_dist = self._manhattan_distance(head, self.food)

        # Update direction
        self.direction = self._get_new_direction(action)
        dx, dy = DIRECTION_DELTAS[self.direction]
        new_head = Point(head.x + dx, head.y + dy)

        # Check collision
        if self._is_collision(new_head):
            self.game_over = True
            reward = self.collision_penalty
            terminated = True
            truncated = False
            obs = self.get_state()
            info = {
                "score": self.score,
                "snake_length": len(self.snake),
                "steps": self.steps,
                "reason": "collision",
            }
            return obs, reward, terminated, truncated, info

        # Move snake
        self.snake.insert(0, new_head)
        curr_dist = self._manhattan_distance(new_head, self.food)

        reward = self.step_penalty

        # Distance shaping reward
        if curr_dist < prev_dist:
            reward += self.closer_reward
        else:
            reward += self.farther_penalty

        # Check food consumption
        if new_head == self.food:
            self.score += 1
            reward += self.food_reward
            self.idle_steps = 0
            self._spawn_food()

            # Win condition: snake occupies entire board
            if len(self.snake) == self.grid_width * self.grid_height:
                self.game_over = True
                terminated = True
                truncated = False
                obs = self.get_state()
                info = {
                    "score": self.score,
                    "snake_length": len(self.snake),
                    "steps": self.steps,
                    "reason": "win_completed_board",
                }
                return obs, reward, terminated, truncated, info
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        # Check starvation / infinite loop timeout
        max_idle = max(100, len(self.snake) * self.max_idle_steps_multiplier)
        if self.idle_steps >= max_idle:
            self.game_over = True
            reward += self.collision_penalty  # Penalty for starving/looping
            terminated = False
            truncated = True
            obs = self.get_state()
            info = {
                "score": self.score,
                "snake_length": len(self.snake),
                "steps": self.steps,
                "reason": "starvation_timeout",
            }
            return obs, reward, terminated, truncated, info

        obs = self.get_state()
        info = {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps": self.steps,
            "reason": "step_ok",
        }
        return obs, reward, False, False, info

    def get_state(self) -> np.ndarray:
        """
        Constructs the 11-dimensional observation state.

        Features:
        [0] Danger straight (1.0 if danger, 0.0 otherwise)
        [1] Danger right
        [2] Danger left
        [3] Direction left
        [4] Direction right
        [5] Direction up
        [6] Direction down
        [7] Food left (food.x < head.x)
        [8] Food right (food.x > head.x)
        [9] Food up (food.y < head.y)
        [10] Food down (food.y > head.y)

        Returns:
            np.ndarray: shape (11,), dtype=np.float32
        """
        head = self.snake[0] if self.snake else Point(0, 0)

        # Precompute next coordinates for 4 absolute directions
        pt_up = Point(head.x, head.y - 1)
        pt_right = Point(head.x + 1, head.y)
        pt_down = Point(head.x, head.y + 1)
        pt_left = Point(head.x - 1, head.y)

        # Danger relative to current direction
        if self.direction == Direction.UP:
            danger_straight = self._is_collision(pt_up)
            danger_right = self._is_collision(pt_right)
            danger_left = self._is_collision(pt_left)
        elif self.direction == Direction.RIGHT:
            danger_straight = self._is_collision(pt_right)
            danger_right = self._is_collision(pt_down)
            danger_left = self._is_collision(pt_up)
        elif self.direction == Direction.DOWN:
            danger_straight = self._is_collision(pt_down)
            danger_right = self._is_collision(pt_left)
            danger_left = self._is_collision(pt_right)
        else:  # LEFT
            danger_straight = self._is_collision(pt_left)
            danger_right = self._is_collision(pt_up)
            danger_left = self._is_collision(pt_down)

        state = [
            # Danger relative
            float(danger_straight),
            float(danger_right),
            float(danger_left),
            # Current absolute direction
            float(self.direction == Direction.LEFT),
            float(self.direction == Direction.RIGHT),
            float(self.direction == Direction.UP),
            float(self.direction == Direction.DOWN),
            # Food relative position
            float(self.food.x < head.x),
            float(self.food.x > head.x),
            float(self.food.y < head.y),
            float(self.food.y > head.y),
        ]

        return np.array(state, dtype=np.float32)
