"""
Publication-Quality Visualization Suite for DQN Snake Agent.
Generates comprehensive dark-themed plots, evaluation comparisons, and gameplay GIFs.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.agent.dqn_agent import DQNAgent
from src.environment.game import SnakeGame
from src.environment.renderer import SnakeRenderer

# Setup Matplotlib Dark Theme
plt.style.use("dark_background")
PALETTE = {
    "bg": "#12141D",
    "card": "#181B26",
    "grid": "#232838",
    "text": "#F1F5F9",
    "primary": "#6366F1",    # Indigo
    "success": "#10B981",    # Emerald
    "danger": "#EF4444",     # Crimson
    "warning": "#F59E0B",    # Amber
    "accent": "#EC4899",     # Pink
    "cyan": "#06B6D4",       # Cyan
}


def setup_figure(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    ax.grid(True, color=PALETTE["grid"], linestyle="--", alpha=0.6, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
        spine.set_linewidth(1.2)
    ax.tick_params(colors=PALETTE["text"], labelsize=10)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    return fig, ax


def plot_reward_curve(metrics_csv: str, output_path: str = "results/figures/01_reward_curve.png") -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = setup_figure((10, 5))

    ax.plot(df["episode"], df["total_reward"], color="#64748B", alpha=0.35, label="Raw Episode Reward")
    ax.plot(df["episode"], df["rolling_reward"], color=PALETTE["cyan"], linewidth=2.2, label="Moving Average (30 ep)")

    ax.set_title("Training Reward Progression per Episode", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Reward", fontsize=11, labelpad=8)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_score_curve(metrics_csv: str, output_path: str = "results/figures/02_score_curve.png") -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = setup_figure((10, 5))

    ax.scatter(df["episode"], df["score"], color=PALETTE["warning"], alpha=0.4, s=18, label="Episode Score")
    ax.plot(df["episode"], df["rolling_score"], color=PALETTE["success"], linewidth=2.4, label="Moving Average Score")

    max_score = df["score"].max()
    max_ep = df.loc[df["score"].idxmax()]["episode"]
    ax.annotate(
        f"Peak Score: {int(max_score)} (Ep {int(max_ep)})",
        xy=(max_ep, max_score),
        xytext=(max_ep - 40 if max_ep > 50 else max_ep + 10, max_score + 1.5),
        arrowprops=dict(facecolor=PALETTE["accent"], shrink=0.08, width=1.5, headwidth=6),
        color=PALETTE["accent"],
        fontweight="bold",
        fontsize=10,
    )

    ax.set_title("Snake Game Score Learning Curve", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Game Score (Apples Eaten)", fontsize=11, labelpad=8)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_episode_length(metrics_csv: str, output_path: str = "results/figures/03_episode_length.png") -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = setup_figure((10, 5))

    ax.plot(df["episode"], df["steps"], color="#94A3B8", alpha=0.35, label="Raw Steps")
    rolling_steps = df["steps"].rolling(window=30, min_periods=1).mean()
    ax.plot(df["episode"], rolling_steps, color=PALETTE["primary"], linewidth=2.2, label="Moving Avg Steps (30 ep)")

    ax.set_title("Survival Episode Length (Steps)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Survival Steps", fontsize=11, labelpad=8)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_epsilon_decay(metrics_csv: str, output_path: str = "results/figures/04_epsilon_decay.png") -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = setup_figure((10, 4.5))

    ax.plot(df["episode"], df["epsilon"], color=PALETTE["accent"], linewidth=2.5, label="Exploration Rate (Epsilon)")
    ax.set_title("Epsilon-Greedy Exploration Decay Schedule", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Epsilon (Exploration Probability)", fontsize=11, labelpad=8)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_training_loss(metrics_csv: str, output_path: str = "results/figures/05_training_loss.png") -> None:
    df = pd.read_csv(metrics_csv)
    fig, ax = setup_figure((10, 5))

    valid_loss = df.dropna(subset=["loss"])
    if not valid_loss.empty:
        ax.plot(valid_loss["episode"], valid_loss["loss"], color="#64748B", alpha=0.35, label="Batch Loss")
        rolling_loss = valid_loss["loss"].rolling(window=30, min_periods=1).mean()
        ax.plot(valid_loss["episode"], rolling_loss, color=PALETTE["danger"], linewidth=2.2, label="Moving Avg Loss")

    ax.set_title("Temporal-Difference (Huber) Training Loss", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Huber Loss", fontsize=11, labelpad=8)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_baseline_comparison(eval_csv: str, output_path: str = "results/figures/06_baseline_comparison.png") -> None:
    df = pd.read_csv(eval_csv)
    fig, ax = setup_figure((9, 6))

    agents = df["agent"].unique()
    colors = [PALETTE["cyan"], PALETTE["danger"], PALETTE["warning"]]

    box_data = [df[df["agent"] == ag]["score"].values for ag in agents]
    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        tick_labels=agents,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(color=PALETTE["text"], linewidth=1.2),
        capprops=dict(color=PALETTE["text"], linewidth=1.2),
        medianprops=dict(color="#FFFFFF", linewidth=2.0),
    )

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
        patch.set_edgecolor(col)

    # Overlay jittered scatter points
    for i, ag in enumerate(agents):
        vals = df[df["agent"] == ag]["score"].values
        jitter = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(jitter, vals, color=colors[i % len(colors)], alpha=0.5, s=25, edgecolor="none")

    ax.set_title("Evaluation Score Distribution: DQN vs Baselines", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Game Score (Apples Eaten)", fontsize=11, labelpad=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_score_distribution(eval_csv: str, output_path: str = "results/figures/07_score_distribution.png") -> None:
    df = pd.read_csv(eval_csv)
    fig, ax = setup_figure((10, 5))

    agents = df["agent"].unique()
    colors = {
        "DQN Agent": PALETTE["cyan"],
        "Random Agent": PALETTE["danger"],
        "Heuristic (Rule-based)": PALETTE["warning"],
    }

    max_s = int(df["score"].max())
    bins = np.arange(0, max_s + 3) - 0.5

    for ag in agents:
        sub = df[df["agent"] == ag]
        c = colors.get(ag, PALETTE["primary"])
        ax.hist(sub["score"], bins=bins, alpha=0.5, label=f"{ag} (mean: {sub['score'].mean():.1f})", color=c, edgecolor=c)

    ax.set_title("Score Histogram across Evaluation Episodes", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Score", fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=11, labelpad=8)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_performance_summary(eval_summary_json: str, output_path: str = "results/figures/08_performance_summary.png") -> None:
    import json
    with open(eval_summary_json, "r") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    headers = ["Agent", "Avg Score", "Median", "Max Score", "Avg Reward", "Avg Steps", "Success (≥15)"]
    rows = []
    for k, v in data.items():
        rows.append([
            v["agent"],
            f"{v['avg_score']:.2f}",
            f"{v['median_score']:.1f}",
            f"{v['max_score']}",
            f"{v['avg_reward']:.2f}",
            f"{v['avg_steps']:.1f}",
            f"{v['success_rate']:.1f}%",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Style table
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_facecolor(PALETTE["primary"])
            cell.set_text_props(color="#FFFFFF", fontweight="bold")
        else:
            cell.set_facecolor(PALETTE["card"] if r % 2 == 0 else PALETTE["bg"])
            cell.set_text_props(color=PALETTE["text"])

    ax.set_title("Benchmark Performance Scorecard Summary", fontsize=14, fontweight="bold", color=PALETTE["text"], pad=20)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_gameplay_demo(
    model_path: str = "models/best_model.pt",
    output_gif: str = "results/figures/09_gameplay_demo.gif",
    output_frame: str = "results/figures/10_gameplay_frame.png",
    seed: int = 42,
    max_steps: int = 600,
) -> None:
    """
    Simulates a trained DQN agent playing a game and records GIF and frame snapshot.
    """
    agent = DQNAgent(state_dim=11, action_dim=3)
    agent.load_checkpoint(model_path)

    env = SnakeGame(grid_width=20, grid_height=20, seed=seed)
    renderer = SnakeRenderer(grid_width=20, grid_height=20, cell_size=24)

    state, _ = env.reset(seed=seed)
    frames: List[Image.Image] = []
    done = False
    step_count = 0

    action_names = ["STRAIGHT", "RIGHT", "LEFT"]

    while not done and step_count < max_steps:
        action, q_vals = agent.select_action(state, explore=False, epsilon=0.0)
        img = renderer.render_frame(
            env,
            episode=1,
            q_values=q_vals,
            action_name=action_names[action],
        )
        frames.append(img)

        # Save middle frame snapshot
        if step_count == 35 or (step_count == 10 and not Path(output_frame).exists()):
            img.save(output_frame)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        step_count += 1

    # Ensure snapshot exists
    if frames and not Path(output_frame).exists():
        frames[min(len(frames) - 1, 20)].save(output_frame)

    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
    renderer.save_gif(frames, output_gif, duration_ms=80)
    print(f"[DEMO] Saved gameplay animation ({len(frames)} frames) to: {output_gif}")
    print(f"[DEMO] Final Game Score: {env.score}, Total Steps: {env.steps}")


def generate_all_plots(
    metrics_csv: str = "results/metrics/training_metrics.csv",
    eval_csv: str = "results/evaluation/evaluation_results.csv",
    eval_summary_json: str = "results/evaluation/evaluation_summary.json",
    model_path: str = "models/best_model.pt",
    figures_dir: str = "results/figures",
) -> None:
    """Generates all mandatory plots and media assets."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("--> Generating visualization suite...")
    plot_reward_curve(metrics_csv, str(fig_dir / "01_reward_curve.png"))
    plot_score_curve(metrics_csv, str(fig_dir / "02_score_curve.png"))
    plot_episode_length(metrics_csv, str(fig_dir / "03_episode_length.png"))
    plot_epsilon_decay(metrics_csv, str(fig_dir / "04_epsilon_decay.png"))
    plot_training_loss(metrics_csv, str(fig_dir / "05_training_loss.png"))
    plot_baseline_comparison(eval_csv, str(fig_dir / "06_baseline_comparison.png"))
    plot_score_distribution(eval_csv, str(fig_dir / "07_score_distribution.png"))
    plot_performance_summary(eval_summary_json, str(fig_dir / "08_performance_summary.png"))
    generate_gameplay_demo(
        model_path=model_path,
        output_gif=str(fig_dir / "09_gameplay_demo.gif"),
        output_frame=str(fig_dir / "10_gameplay_frame.png"),
    )
    print(f"--> All 10 visual assets successfully generated in: {fig_dir}")
