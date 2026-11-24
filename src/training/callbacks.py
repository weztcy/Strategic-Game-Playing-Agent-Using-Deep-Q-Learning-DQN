"""
Training callbacks and logger for monitoring DQN progress.
"""

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class TrainingLogger:
    """
    Logs episode metrics, computes moving averages, and persists history to CSV.
    """

    def __init__(self, log_dir: str = "results/metrics", window_size: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_metrics.csv"
        self.window_size = window_size

        self.records: List[Dict] = []
        self.recent_scores = deque(maxlen=window_size)
        self.recent_rewards = deque(maxlen=window_size)

    def log_episode(
        self,
        episode: int,
        score: int,
        total_reward: float,
        steps: int,
        epsilon: float,
        loss: Optional[float],
        duration_sec: float,
    ) -> Dict:
        """Records an episode's metrics."""
        self.recent_scores.append(score)
        self.recent_rewards.append(total_reward)

        rolling_score = float(sum(self.recent_scores) / len(self.recent_scores))
        rolling_reward = float(sum(self.recent_rewards) / len(self.recent_rewards))

        record = {
            "episode": episode,
            "score": score,
            "rolling_score": round(rolling_score, 3),
            "total_reward": round(total_reward, 3),
            "rolling_reward": round(rolling_reward, 3),
            "steps": steps,
            "epsilon": round(epsilon, 4),
            "loss": round(loss, 6) if loss is not None else float("nan"),
            "duration_sec": round(duration_sec, 3),
        }
        self.records.append(record)
        return record

    def save_csv(self) -> Path:
        """Saves all logged records to CSV."""
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)
        return self.csv_path
