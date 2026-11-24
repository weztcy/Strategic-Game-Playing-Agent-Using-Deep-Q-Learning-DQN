"""
Rigorous Evaluation Pipeline for DQN Agent vs Baselines.
Runs multi-seed benchmarking across identical environment configurations.
Generates comprehensive statistical summaries and CSV outputs.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

from src.agent.dqn_agent import DQNAgent
from src.environment.game import SnakeGame
from src.evaluation.baselines import HeuristicAgent, RandomAgent


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_agent(
    agent,
    agent_name: str,
    episodes: int = 100,
    seeds: List[int] = None,
    config: dict = None,
    success_threshold: int = 15,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates an agent on SnakeGame across multiple episodes and seeds.
    """
    if seeds is None:
        seeds = [42, 101, 202, 303, 404]

    env_cfg = config.get("environment", {})
    reward_cfg = config.get("rewards", {})

    records = []
    episodes_per_seed = int(np.ceil(episodes / len(seeds)))
    ep_count = 0

    print(f"--> Evaluating {agent_name} over {episodes} episodes across seeds {seeds}...")

    start_time = time.time()

    for s_idx, seed in enumerate(seeds):
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

        for i in range(episodes_per_seed):
            if ep_count >= episodes:
                break
            ep_count += 1
            ep_seed = seed * 1000 + i
            state, _ = env.reset(seed=ep_seed)
            ep_reward = 0.0
            done = False
            ep_start = time.time()

            while not done:
                if isinstance(agent, HeuristicAgent):
                    action, _ = agent.select_action_with_game(env)
                elif isinstance(agent, DQNAgent):
                    action, _ = agent.select_action(state, explore=False, epsilon=0.0)
                else:  # RandomAgent
                    action, _ = agent.select_action(state, explore=True)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                ep_reward += reward

            ep_duration = time.time() - ep_start
            success = env.score >= success_threshold

            records.append(
                {
                    "agent": agent_name,
                    "seed": seed,
                    "episode": ep_count,
                    "score": env.score,
                    "total_reward": round(ep_reward, 3),
                    "steps": env.steps,
                    "success": success,
                    "duration_sec": round(ep_duration, 4),
                    "reason": info.get("reason", "unknown"),
                }
            )

    total_time = time.time() - start_time
    df = pd.DataFrame(records)

    summary = {
        "agent": agent_name,
        "episodes": len(df),
        "avg_score": round(float(df["score"].mean()), 3),
        "median_score": round(float(df["score"].median()), 3),
        "std_score": round(float(df["score"].std()), 3),
        "min_score": int(df["score"].min()),
        "max_score": int(df["score"].max()),
        "avg_reward": round(float(df["total_reward"].mean()), 3),
        "avg_steps": round(float(df["steps"].mean()), 3),
        "success_rate": round(float(df["success"].mean() * 100.0), 2),
        "avg_duration_sec": round(float(df["duration_sec"].mean()), 4),
        "total_eval_time_sec": round(total_time, 2),
    }

    return df, summary


def run_evaluation(
    model_path: str = "models/best_model.pt",
    config_path: str = "config/config.yaml",
    episodes: int = 100,
    output_dir: str = "results/evaluation",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Runs full comparative evaluation between DQN, Random Agent, and Heuristic Agent.
    """
    config = load_config(config_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    seeds = config.get("evaluation", {}).get("seeds", [42, 101, 202, 303, 404])
    success_threshold = config.get("evaluation", {}).get("success_score_threshold", 15)

    # 1. DQN Agent
    agent_cfg = config.get("agent", {})
    dqn_agent = DQNAgent(
        state_dim=agent_cfg.get("state_dim", 11),
        action_dim=agent_cfg.get("action_dim", 3),
        hidden_dims=tuple(agent_cfg.get("hidden_dims", [256, 256])),
        use_double_dqn=True,
    )
    model_file = Path(model_path)
    if model_file.exists():
        print(f"Loading trained weights from {model_file}...")
        meta = dqn_agent.load_checkpoint(model_file)
        print(f"Loaded checkpoint metadata: {meta}")
    else:
        print(f"[WARNING] Model checkpoint {model_file} not found. Evaluating uninitialized DQN.")

    # 2. Random Agent
    random_agent = RandomAgent(action_dim=agent_cfg.get("action_dim", 3), seed=seeds[0])

    # 3. Heuristic Agent
    heuristic_agent = HeuristicAgent(seed=seeds[0])

    agents = [
        (dqn_agent, "DQN Agent"),
        (random_agent, "Random Agent"),
        (heuristic_agent, "Heuristic (Rule-based)"),
    ]

    all_dfs = []
    summaries = {}

    for ag, name in agents:
        df, summary = evaluate_agent(
            ag,
            name,
            episodes=episodes,
            seeds=seeds,
            config=config,
            success_threshold=success_threshold,
        )
        all_dfs.append(df)
        summaries[name] = summary

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv = out_path / "evaluation_results.csv"
    combined_df.to_csv(combined_csv, index=False)

    summary_json = out_path / "evaluation_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    # Format Markdown Table for Console Output
    print("\n" + "=" * 80)
    print("                     EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(list(summaries.values()))
    display_cols = [
        "agent",
        "avg_score",
        "median_score",
        "std_score",
        "min_score",
        "max_score",
        "avg_reward",
        "avg_steps",
        "success_rate",
    ]
    print(summary_df[display_cols].to_markdown(index=False))
    print("=" * 80)
    print(f"Detailed results saved to: {combined_csv}")
    print(f"Summary JSON saved to: {summary_json}\n")

    return combined_df, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN Agent against Baselines")
    parser.add_argument("--model", type=str, default="models/best_model.pt", help="Path to trained model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--outdir", type=str, default="results/evaluation", help="Output directory")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        config_path=args.config,
        episodes=args.episodes,
        output_dir=args.outdir,
    )
