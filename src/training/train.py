"""
Main Training Script for DQN Snake Agent.
Trains policy network with experience replay and target network updates.
"""

import argparse
from pathlib import Path
import time
import numpy as np
import yaml

from src.agent.dqn_agent import DQNAgent
from src.environment.game import SnakeGame
from src.training.callbacks import TrainingLogger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_dqn(
    episodes: int = 600,
    seed: int = 42,
    config_path: str = "config/config.yaml",
    save_dir: str = "models",
    log_dir: str = "results/metrics",
    smoke_test: bool = False,
) -> DQNAgent:
    """
    Executes training loop for DQN Agent on SnakeGame.
    """
    cfg = load_config(config_path)

    env_cfg = cfg.get("environment", {})
    agent_cfg = cfg.get("agent", {})
    reward_cfg = cfg.get("rewards", {})

    if smoke_test:
        episodes = cfg.get("training", {}).get("smoke_test_episodes", 15)

    print(f"=== Starting DQN Training (Episodes: {episodes}, Seed: {seed}, Smoke: {smoke_test}) ===")

    # Setup directories
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    env = SnakeGame(
        grid_width=env_cfg.get("grid_width", 20),
        grid_height=env_cfg.get("grid_height", 20),
        max_idle_steps_multiplier=env_cfg.get("max_idle_steps_multiplier", 100),
        food_reward=reward_cfg.get("food_reward", 10.0),
        collision_penalty=reward_cfg.get("collision_penalty", -10.0),
        closer_reward=reward_cfg.get("closer_reward", 0.1),
        farther_penalty=reward_cfg.get("farther_penalty", -0.15),
        step_penalty=reward_cfg.get("step_penalty", -0.01),
        seed=seed,
    )

    # Initialize agent
    agent = DQNAgent(
        state_dim=agent_cfg.get("state_dim", 11),
        action_dim=agent_cfg.get("action_dim", 3),
        hidden_dims=tuple(agent_cfg.get("hidden_dims", [256, 256])),
        learning_rate=float(agent_cfg.get("learning_rate", 0.0005)),
        gamma=float(agent_cfg.get("gamma", 0.99)),
        epsilon_start=float(agent_cfg.get("epsilon_start", 1.0)),
        epsilon_min=float(agent_cfg.get("epsilon_min", 0.01)),
        epsilon_decay=float(agent_cfg.get("epsilon_decay", 0.995)),
        buffer_capacity=int(agent_cfg.get("buffer_capacity", 100000)),
        batch_size=int(agent_cfg.get("batch_size", 64)),
        target_update_freq=int(agent_cfg.get("target_update_freq", 500)),
        grad_clip=float(agent_cfg.get("grad_clip", 1.0)),
        use_double_dqn=True,
        seed=seed,
    )

    logger = TrainingLogger(log_dir=log_dir, window_size=30)

    best_rolling_score = -1.0
    highest_score = 0
    start_total_time = time.time()

    for ep in range(1, episodes + 1):
        ep_start_time = time.time()
        # Seed per episode for controlled variety
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        losses = []

        done = False
        while not done:
            action, _ = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        ep_duration = time.time() - ep_start_time
        avg_loss = float(np.mean(losses)) if losses else None

        record = logger.log_episode(
            episode=ep,
            score=env.score,
            total_reward=ep_reward,
            steps=env.steps,
            epsilon=agent.epsilon,
            loss=avg_loss,
            duration_sec=ep_duration,
        )

        if env.score > highest_score:
            highest_score = env.score

        # Save Best Model based on rolling score (after warm-up of 20 episodes)
        if ep >= 20 and record["rolling_score"] > best_rolling_score:
            best_rolling_score = record["rolling_score"]
            agent.save_checkpoint(
                save_path / "best_model.pt",
                metadata={
                    "episode": ep,
                    "rolling_score": best_rolling_score,
                    "highest_score": highest_score,
                    "epsilon": agent.epsilon,
                },
            )

        # Periodic checkpoint
        if ep % cfg.get("training", {}).get("checkpoint_interval", 100) == 0:
            agent.save_checkpoint(
                checkpoints_dir / f"checkpoint_ep{ep}.pt",
                metadata={"episode": ep, "score": env.score, "rolling_score": record["rolling_score"]},
            )

        # Console logging
        if ep % 20 == 0 or ep == 1 or ep == episodes:
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(
                f"Ep {ep:4d}/{episodes} | "
                f"Score: {env.score:2d} (Best: {highest_score:2d}, RollAvg: {record['rolling_score']:.2f}) | "
                f"Reward: {ep_reward:7.2f} | "
                f"Steps: {env.steps:4d} | "
                f"Loss: {loss_str} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Time: {ep_duration:.2f}s"
            )

    total_training_time = time.time() - start_total_time
    csv_file = logger.save_csv()
    print(f"\n[DONE] Training finished in {total_training_time:.2f}s.")
    print(f"Metrics saved to: {csv_file}")
    print(f"Best rolling score: {best_rolling_score:.2f} | All-time highest score: {highest_score}")

    # Ensure best_model exists even if training was short
    best_model_file = save_path / "best_model.pt"
    if not best_model_file.exists():
        agent.save_checkpoint(
            best_model_file,
            metadata={"episode": episodes, "score": env.score, "rolling_score": best_rolling_score},
        )

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent on Snake")
    parser.add_argument("--episodes", type=int, default=600, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    train_dqn(
        episodes=args.episodes,
        seed=args.seed,
        config_path=args.config,
        smoke_test=args.smoke_test,
    )
