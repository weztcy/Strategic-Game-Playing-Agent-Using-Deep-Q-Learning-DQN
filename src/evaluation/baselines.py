"""
Baseline Agents for Comparative Evaluation.
Includes Random Agent and Heuristic Obstacle-Avoidance Agent.
"""

from typing import Optional, Tuple
import numpy as np

from src.environment.game import Action, Direction, DIRECTION_DELTAS, Point, SnakeGame


class RandomAgent:
    """
    Uniform Random Agent: randomly picks a valid action from [0, 1, 2].
    """

    def __init__(self, action_dim: int = 3, seed: int = 42):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        action = int(self.rng.integers(0, self.action_dim))
        dummy_q = np.zeros(self.action_dim, dtype=np.float32)
        return action, dummy_q


class HeuristicAgent:
    """
    Rule-based Heuristic Agent:
    1. Evaluates all 3 candidate moves (Straight, Turn Right, Turn Left).
    2. Filters out actions that lead to immediate collision with wall or body.
    3. Among safe actions, chooses the one minimizing Manhattan distance to food.
    4. If no safe action exists, defaults to Straight.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def select_action_with_game(
        self,
        game: SnakeGame,
    ) -> Tuple[int, np.ndarray]:
        head = game.snake[0]
        food = game.food
        curr_dir = game.direction

        candidate_actions = [Action.STRAIGHT, Action.TURN_RIGHT, Action.TURN_LEFT]
        safe_actions = []

        for act in candidate_actions:
            new_dir = game._get_new_direction(act)
            dx, dy = DIRECTION_DELTAS[new_dir]
            new_head = Point(head.x + dx, head.y + dy)

            # Check collision
            if not game._is_collision(new_head):
                dist = abs(new_head.x - food.x) + abs(new_head.y - food.y)
                safe_actions.append((act, dist))

        dummy_q = np.zeros(3, dtype=np.float32)

        if not safe_actions:
            # Trapped with no safe move; default to straight
            return int(Action.STRAIGHT), dummy_q

        # Pick safe action that minimizes distance to food
        safe_actions.sort(key=lambda item: item[1])
        best_act = safe_actions[0][0]
        dummy_q[best_act] = 1.0
        return int(best_act), dummy_q

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = False,
        epsilon: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Fallback selector using state vector when game object is not directly passed.
        State indices:
        [0] danger_straight, [1] danger_right, [2] danger_left
        [7] food_left, [8] food_right, [9] food_up, [10] food_down
        """
        danger_s, danger_r, danger_l = state[0], state[1], state[2]
        safe_acts = []
        if danger_s == 0.0:
            safe_acts.append(int(Action.STRAIGHT))
        if danger_r == 0.0:
            safe_acts.append(int(Action.TURN_RIGHT))
        if danger_l == 0.0:
            safe_acts.append(int(Action.TURN_LEFT))

        dummy_q = np.zeros(3, dtype=np.float32)
        if not safe_acts:
            return int(Action.STRAIGHT), dummy_q

        # Pick safe action randomly among candidates
        chosen = int(self.rng.choice(safe_acts))
        dummy_q[chosen] = 1.0
        return chosen, dummy_q
