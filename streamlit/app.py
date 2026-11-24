"""
Interactive Streamlit Application for DQN Snake Agent.
Provides modern dark-theme dashboard, live gameplay simulation, training analytics,
baseline benchmarks, and neural network state inspector.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.agent.dqn_agent import DQNAgent
from src.environment.game import Action, Direction, SnakeGame
from src.environment.renderer import SnakeRenderer
from src.evaluation.baselines import HeuristicAgent, RandomAgent

# Page Configuration
st.set_page_config(
    page_title="Strategic DQN Game Playing Agent | Snake AI",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Theme Custom Styling
DARK_CSS = """
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0E1117;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E232E 0%, #161A22 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Headers & Dividers */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    hr {
        border-color: #30363D !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #30363D;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 8px 8px 0 0;
        color: #94A3B8;
        padding: 10px 20px;
        border: 1px solid #30363D;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border-color: #38BDF8 !important;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


def get_paths():
    base = Path(__file__).parent
    return {
        "model": base / "models" / "best_model.pt",
        "metrics": base / "results" / "metrics" / "training_metrics.csv",
        "eval_csv": base / "results" / "evaluation" / "evaluation_results.csv",
        "eval_json": base / "results" / "evaluation" / "evaluation_summary.json",
        "figures": base / "results" / "figures",
    }


PATHS = get_paths()

# Sidebar Navigation
st.sidebar.title("🐍 DQN Snake AI")
st.sidebar.caption("Autonomous Strategic Game Playing Agent")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard Overview",
        "🎮 Live Game Simulation",
        "📈 Training Analytics",
        "⚖️ Baseline Benchmarks",
        "🔍 Neural State Inspector",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
model_exists = PATHS["model"].exists()
st.sidebar.markdown(f"**Model Status:** {'🟢 Best Model Loaded' if model_exists else '🟡 Model Not Found'}")
st.sidebar.markdown(f"**Device:** `CPU (PyTorch)`")
st.sidebar.markdown(f"**Action Space:** `Discrete(3) Relative`")
st.sidebar.markdown(f"**State Features:** `11-Dim Binary Vector`")


# ==============================================================================
# TAB 1: DASHBOARD OVERVIEW
# ==============================================================================
if page == "📊 Dashboard Overview":
    st.title("Strategic Game Playing Agent Using Deep Q-Learning (DQN)")
    st.markdown(
        """
        Implementation of an autonomous Reinforcement Learning agent trained to master **Snake** 
        from raw trial-and-error interactions. The agent utilizes **Double Deep Q-Networks (DDQN)** 
        with experience replay, target network synchronization, and relative action spaces.
        """
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Algorithm", "Double DQN")
    with col2:
        st.metric("Grid Dimensions", "20 × 20 (400 Cells)")
    with col3:
        st.metric("Observation Space", "11 Features")
    with col4:
        st.metric("Action Space", "3 Relative Actions")

    st.markdown("### 🏆 Performance Snapshot")
    if PATHS["eval_json"].exists():
        with open(PATHS["eval_json"], "r") as f:
            eval_data = json.load(f)
        dqn_data = eval_data.get("DQN Agent", {})
        rnd_data = eval_data.get("Random Agent", {})
        heur_data = eval_data.get("Heuristic (Rule-based)", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DQN Mean Score", f"{dqn_data.get('avg_score', 'N/A')}", delta=f"+{round(dqn_data.get('avg_score', 0) - rnd_data.get('avg_score', 0), 2)} vs Random")
        c2.metric("DQN Peak Score", f"{dqn_data.get('max_score', 'N/A')}")
        c3.metric("Survival Steps (Mean)", f"{dqn_data.get('avg_steps', 'N/A')}")
        c4.metric("Success Rate (≥15)", f"{dqn_data.get('success_rate', 'N/A')}%")
    else:
        st.info("Evaluation metrics will appear once training and evaluation pipelines complete.")

    st.markdown("### 🏗️ Architecture & Reinforcement Learning Design")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            r"""
            **Markov Decision Process (MDP):**
            - **State $S \in \mathbb{R}^{11}$**: Danger [Straight, Right, Left], Direction [Left, Right, Up, Down], Food Proximity [Left, Right, Up, Down].
            - **Action $A \in \{0, 1, 2\}$**: [0] Straight, [1] Turn Right, [2] Turn Left. Eliminates 180° immediate self-reversals.
            - **Reward $R$**: $+10.0$ for food, $-10.0$ for death, $+0.1$ for closing distance, $-0.15$ for moving away, $-0.01$ step penalty.
            - **Termination**: Wall collision, self-collision, or starvation timeout ($100 \times \text{length}$).
            """
        )
    with col_b:
        st.markdown(
            r"""
            **Deep Q-Network Specifications:**
            - **Network Architecture**: MLP with 2 hidden layers of 256 units + ReLU activations.
            - **Experience Replay**: Circular buffer capacity $100,000$, mini-batch size $64$.
            - **Target Network**: Synced every $500$ update steps via parameter copying.
            - **Double DQN Bellman Loss**: Huber loss (Smooth L1) with gradient norm clipping $1.0$.
            - **Optimizer**: Adam ($\text{lr} = 0.0005$, $\gamma = 0.99$).
            """
        )

    if (PATHS["figures"] / "09_gameplay_demo.gif").exists():
        st.markdown("### 🎥 Autonomous Gameplay Demo")
        st.image(str(PATHS["figures"] / "09_gameplay_demo.gif"), caption="Trained Autonomous DQN Agent playing Snake", use_container_width=True)


# ==============================================================================
# TAB 2: LIVE GAME SIMULATION
# ==============================================================================
elif page == "🎮 Live Game Simulation":
    st.title("🎮 Live Gameplay Demonstration")
    st.markdown("Watch the trained autonomous DQN agent or baseline agents play live on a randomized grid.")

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        agent_type = st.selectbox("Select Agent Policy", ["Trained DQN Agent", "Heuristic (Rule-based)", "Random Agent"])
        sim_seed = st.number_input("Game Random Seed", min_value=0, max_value=999999, value=42)
        sim_speed = st.slider("Animation Delay (seconds)", min_value=0.01, max_value=0.30, value=0.05, step=0.01)
        max_steps = st.slider("Max Episode Steps", min_value=50, max_value=1000, value=400, step=50)

        start_btn = st.button("▶️ Run Simulation")

    with sim_col2:
        game_canvas = st.empty()
        status_bar = st.empty()
        q_bar = st.empty()

    if start_btn:
        env = SnakeGame(grid_width=20, grid_height=20, seed=sim_seed)
        renderer = SnakeRenderer(grid_width=20, grid_height=20, cell_size=24)

        if agent_type == "Trained DQN Agent":
            agent = DQNAgent(state_dim=11, action_dim=3)
            if PATHS["model"].exists():
                agent.load_checkpoint(PATHS["model"])
        elif agent_type == "Heuristic (Rule-based)":
            agent = HeuristicAgent(seed=sim_seed)
        else:
            agent = RandomAgent(action_dim=3, seed=sim_seed)

        state, _ = env.reset(seed=sim_seed)
        done = False
        action_names = ["STRAIGHT", "TURN RIGHT", "TURN LEFT"]

        for step in range(max_steps):
            if isinstance(agent, HeuristicAgent):
                action, q_vals = agent.select_action_with_game(env)
            elif isinstance(agent, DQNAgent):
                action, q_vals = agent.select_action(state, explore=False, epsilon=0.0)
            else:
                action, q_vals = agent.select_action(state, explore=True)

            img = renderer.render_frame(env, episode=1, q_values=q_vals, action_name=action_names[action])
            game_canvas.image(img, use_container_width=False)

            status_bar.info(
                f"**Step:** {env.steps} | **Score:** {env.score} | **Snake Length:** {len(env.snake)} | **Action:** `{action_names[action]}`"
            )

            if isinstance(q_vals, np.ndarray) and len(q_vals) == 3:
                q_bar.caption(f"**Q-Values:** Straight: `{q_vals[0]:.2f}` | Right: `{q_vals[1]:.2f}` | Left: `{q_vals[2]:.2f}`")

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

            time.sleep(sim_speed)
            if done:
                status_bar.warning(f"🏁 Episode Terminated. Final Score: **{env.score}**, Total Steps: **{env.steps}**, Reason: `{info.get('reason')}`")
                break
    else:
        init_env = SnakeGame(grid_width=20, grid_height=20, seed=int(sim_seed))
        init_renderer = SnakeRenderer(grid_width=20, grid_height=20, cell_size=24)
        init_env.reset(seed=int(sim_seed))
        init_img = init_renderer.render_frame(init_env, episode=1, action_name="READY")
        game_canvas.image(init_img, caption="Snake Game Arena (Ready to Start)", use_container_width=True)
        status_bar.info("Press **▶️ Run Simulation** to watch the selected agent play live in real-time.")
        q_bar.caption("Q-Values will be displayed dynamically at each decision step.")


# ==============================================================================
# TAB 3: TRAINING ANALYTICS
# ==============================================================================
elif page == "📈 Training Analytics":
    st.title("📈 Training Dynamics & Loss Progression")
    st.markdown("Comprehensive visual analysis of the policy training trajectory.")

    if PATHS["metrics"].exists():
        df_metrics = pd.read_csv(PATHS["metrics"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episodes Trained", len(df_metrics))
        c2.metric("Max Training Score", int(df_metrics["score"].max()))
        c3.metric("Final Rolling Score", f"{df_metrics['rolling_score'].iloc[-1]:.2f}")
        c4.metric("Final Epsilon", f"{df_metrics['epsilon'].iloc[-1]:.4f}")

        st.markdown("### 📊 Learning Curves")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            p1 = PATHS["figures"] / "01_reward_curve.png"
            if p1.exists():
                st.image(str(p1), caption="Total Reward per Episode (Raw & Moving Average)", use_container_width=True)
            p3 = PATHS["figures"] / "03_episode_length.png"
            if p3.exists():
                st.image(str(p3), caption="Survival Duration (Steps per Episode)", use_container_width=True)
        with f_col2:
            p2 = PATHS["figures"] / "02_score_curve.png"
            if p2.exists():
                st.image(str(p2), caption="Game Score Progression (Apples Eaten)", use_container_width=True)
            p5 = PATHS["figures"] / "05_training_loss.png"
            if p5.exists():
                st.image(str(p5), caption="Temporal-Difference Loss", use_container_width=True)

        p4 = PATHS["figures"] / "04_epsilon_decay.png"
        if p4.exists():
            st.image(str(p4), caption="Epsilon Exploration Decay Schedule", use_container_width=True)

        with st.expander("📄 View Raw Training Log Table"):
            st.dataframe(df_metrics.tail(100), use_container_width=True)
    else:
        st.warning("Training log not found at `results/metrics/training_metrics.csv`.")


# ==============================================================================
# TAB 4: BASELINE BENCHMARKS
# ==============================================================================
elif page == "⚖️ Baseline Benchmarks":
    st.title("⚖️ Baseline Evaluation & Comparative Benchmarks")
    st.markdown(
        "Rigorous comparative evaluation of the trained **DQN Agent** against **Uniform Random Agent** and **Heuristic Greedy Agent** across multiple independent random seeds."
    )

    if PATHS["eval_json"].exists() and PATHS["eval_csv"].exists():
        with open(PATHS["eval_json"], "r") as f:
            eval_summary = json.load(f)

        st.markdown("### 📋 Statistical Summary Table")
        df_summary = pd.DataFrame(list(eval_summary.values()))
        st.dataframe(
            df_summary[
                ["agent", "episodes", "avg_score", "median_score", "std_score", "max_score", "avg_reward", "avg_steps", "success_rate"]
            ],
            use_container_width=True,
        )

        st.markdown("### 📊 Distributional & Boxplot Comparisons")
        c1, c2 = st.columns(2)
        with c1:
            p6 = PATHS["figures"] / "06_baseline_comparison.png"
            if p6.exists():
                st.image(str(p6), caption="Score Distribution Boxplots", use_container_width=True)
        with c2:
            p7 = PATHS["figures"] / "07_score_distribution.png"
            if p7.exists():
                st.image(str(p7), caption="Score Frequency Histograms", use_container_width=True)

        p8 = PATHS["figures"] / "08_performance_summary.png"
        if p8.exists():
            st.image(str(p8), caption="Comprehensive Metric Scorecard", use_container_width=True)
    else:
        st.warning("Evaluation results not found at `results/evaluation/`.")


# ==============================================================================
# TAB 5: NEURAL STATE INSPECTOR
# ==============================================================================
elif page == "🔍 Neural State Inspector":
    st.title("🔍 Neural State & Q-Value Inspector")
    st.markdown("Interactive tool to inspect the 11 observation features and analyze the network's predicted Q-values.")

    agent = DQNAgent(state_dim=11, action_dim=3)
    if PATHS["model"].exists():
        agent.load_checkpoint(PATHS["model"])

    col_cfg, col_vis = st.columns([1, 1])

    with col_cfg:
        st.subheader("Simulate Observation Features")
        d_straight = st.checkbox("Danger Straight", value=False)
        d_right = st.checkbox("Danger Right", value=False)
        d_left = st.checkbox("Danger Left", value=False)

        st.markdown("**Current Heading Direction:**")
        direction_choice = st.radio("Direction", ["RIGHT", "LEFT", "UP", "DOWN"], horizontal=True)

        st.markdown("**Relative Food Position:**")
        f_left = st.checkbox("Food is to Left (x < head.x)", value=False)
        f_right = st.checkbox("Food is to Right (x > head.x)", value=True)
        f_up = st.checkbox("Food is Above (y < head.y)", value=False)
        f_down = st.checkbox("Food is Below (y > head.y)", value=False)

        state_vector = np.array(
            [
                float(d_straight),
                float(d_right),
                float(d_left),
                float(direction_choice == "LEFT"),
                float(direction_choice == "RIGHT"),
                float(direction_choice == "UP"),
                float(direction_choice == "DOWN"),
                float(f_left),
                float(f_right),
                float(f_up),
                float(f_down),
            ],
            dtype=np.float32,
        )

    with col_vis:
        st.subheader("Model Decision & Q-Values")
        action, q_vals = agent.select_action(state_vector, explore=False, epsilon=0.0)
        action_labels = ["STRAIGHT (0)", "TURN RIGHT (1)", "TURN LEFT (2)"]

        q_df = pd.DataFrame(
            {
                "Action": action_labels,
                "Q-Value": q_vals,
                "Is Selected": [i == action for i in range(3)],
            }
        )

        st.bar_chart(q_df.set_index("Action")["Q-Value"])

        st.success(f"**Policy Action Selected:** `{action_labels[action]}` with Q = `{q_vals[action]:.3f}`")
        st.json(
            {
                "state_vector_11dim": state_vector.tolist(),
                "q_straight": float(q_vals[0]),
                "q_right": float(q_vals[1]),
                "q_left": float(q_vals[2]),
                "argmax_action": int(action),
            }
        )
