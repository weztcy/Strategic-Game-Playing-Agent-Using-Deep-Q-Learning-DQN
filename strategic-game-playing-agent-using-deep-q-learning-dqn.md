# Portfolio Project Data — Strategic Game Playing Agent Using Deep Q-Learning (DQN)

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/strategic_game_playing_agent/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image11.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "strategic-game-playing-agent-using-deep-q-learning-dqn",
  "code": "PRO-31 // ML-16",
  "title": "Strategic Game Playing Agent Using Deep Q-Learning (DQN)",
  "category": "machine-learning",
  "categoryLabel": "Reinforcement Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Strategic-Game-Playing-Agent-Using-Deep-Q-Learning-DQN",
  "license": "MIT License",
  "banner": "/images/projects/strategic_game_playing_agent/banner.webp",
  "summary": "Autonomous Double DQN agent trained on a 20x20 Snake MDP, reaching 29.58 mean score and 60 peak apples across 100 multi-seed evaluation episodes.",
  "projectDescription": "An end-to-end Reinforcement Learning pipeline built with PyTorch that trains a Double Deep Q-Network agent to master a 20x20 Snake Markov Decision Process from trial-and-error interactions, featuring experience replay, multi-seed baseline benchmarking, and a dark-theme Streamlit analytics dashboard.",
  "architectureSummary": "20x20 SnakeGame MDP -> 11-Dim Binary State -> ReplayBuffer (100K) -> Double DQN Policy/Target MLPs -> Huber Loss & Adam -> Multi-Seed Evaluation & Streamlit Inspector",
  "techStack": [
    "Python",
    "PyTorch",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Pillow",
    "PyYAML",
    "Streamlit",
    "Pytest",
    "Tabulate"
  ],
  "telemetryMatrix": {
    "projectType": "Deep Reinforcement Learning Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web Dashboard",
    "primaryDomain": "Sequential Game Playing & Control",
    "architecturePipeline": "MDP Env -> Double DQN -> Eval & UI",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Double DQN Control Agent",
    "learningParadigm": "Reinforcement Learning",
    "modelDomain": "Autonomous Grid Game Playing",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Double DQN Policy",
      "Multi-Seed Benchmarking",
      "Live Q-Value Inspector"
    ]
  },
  "mlProblemContext": "Snake is a dynamic constrained pathfinding problem where every apple eaten extends the agent's body and reshapes obstacle topology across the 20x20 grid. Myopic greedy policies often trap themselves in dead-end loops, requiring a long-horizon value-based agent to balance food pursuit and survival.",
  "mlDatasetOverview": {
    "profileBadge": "Online MDP Experience",
    "source": "SnakeGame Custom MDP Environment & Logged CSV/JSON Telemetry",
    "modality": "Dynamic MDP State & Reward Tensors",
    "scale": "500 Training Episodes (36,696 Steps) / 100 Eval Episodes per Agent",
    "inputShape": "11-Dimensional Binary Float32 Vector",
    "dataFormat": "NumPy Float32 Tensors / CSV / JSON",
    "description": "Experience tuples (state, action, reward, next_state, done) are generated online during interaction and sampled uniformly from a 100,000-transition cyclic replay buffer, while training and multi-seed evaluation trajectories are persisted in CSV and JSON files.",
    "provenanceNote": "Self-contained deterministic MDP simulator with reproducible NumPy and PyTorch seeding."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "11-Dim State & 3-Action Space",
    "spaceNote": "Egocentric local danger perception paired with global heading and relative food quadrant flags.",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "Action-Value Estimates Q(s, a) & Optimal Relative Steering Action",
      "type": "3-Dimensional Continuous Q-Value Vector (Discrete Argmax Policy)",
      "distribution": "Actions: 0 (STRAIGHT), 1 (TURN_RIGHT), 2 (TURN_LEFT); Reward range per step [-10.16, +10.10]"
    },
    "inputSignals": [
      {
        "name": "Local Danger Flags (danger_straight, danger_right, danger_left)",
        "type": "Binary Float32 (3 dims)",
        "importanceOrRole": "Critical Collision Avoidance",
        "description": "Indicates whether the immediate cell straight ahead, to the right, or to the left relative to the snake's heading is a wall or body segment."
      },
      {
        "name": "Current Heading Flags (dir_left, dir_right, dir_up, dir_down)",
        "type": "One-Hot Float32 (4 dims)",
        "importanceOrRole": "Global Orientation Context",
        "description": "One-hot encoding of the snake's active compass direction on the 20x20 grid."
      },
      {
        "name": "Relative Food Quadrant Flags (food_left, food_right, food_up, food_down)",
        "type": "Binary Float32 (4 dims)",
        "importanceOrRole": "Primary Goal Guidance",
        "description": "Encodes the horizontal and vertical coordinate offset of the target apple relative to the snake head."
      }
    ],
    "dataCharacteristicsSummary": "All 11 state signals are normalized binary float32 indicators in {0.0, 1.0}, eliminating feature scaling skew while preserving Markovian local obstacle awareness. Relative action encoding removes invalid 180-degree self-reversals so every action remains physically valid."
  },
  "mlLearningApproach": {
    "paradigm": "Model-Free Value-Based Reinforcement Learning (Double DQN)",
    "methodologyDescription": "The agent approximates Q*(s, a) using a 3-layer MLP policy network decoupled from a periodically synchronized target network via Double DQN target selection. Uniform experience replay breaks temporal correlation across consecutive frames while epsilon-greedy decay transitions exploration to exploitation.",
    "justification": "Unlike myopic Manhattan-distance heuristics that get trapped in self-made pockets, Double DQN with a 0.99 discount factor learns long-horizon perimeter routing while avoiding standard DQN maximization bias."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Egocentric State Vectorization",
      "description": "Convert raw grid coordinates, body segments, and food location into a compact 11-dimensional binary float32 observation vector per step.",
      "badge": "State Encoding"
    },
    {
      "step": "02",
      "title": "Relative Action Mapping",
      "description": "Map discrete actions {0, 1, 2} to Straight, 90-degree clockwise, and 90-degree counter-clockwise turns to prevent fatal reverse collisions.",
      "badge": "Action Space"
    },
    {
      "step": "03",
      "title": "Asymmetric Distance Reward Shaping",
      "description": "Combine +10.0 food rewards and -10.0 collision penalties with +0.10 closer, -0.15 farther, and -0.01 step penalties plus starvation timeouts.",
      "badge": "Reward Design"
    },
    {
      "step": "04",
      "title": "Experience Replay Sampling",
      "description": "Store up to 100,000 transition tuples in a cyclic deque buffer and sample decorrelated mini-batches of 64 transitions for SGD updates.",
      "badge": "Replay Buffer"
    }
  ],
  "mlDataPreparationNote": "Training uses seed 42 with episode offsets (42 + ep), whereas comparative evaluation runs strictly on 5 independent seeds [42, 101, 202, 303, 404] with exploration disabled (epsilon = 0.0).",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Double Deep Q-Network (DDQN)",
    "architectureSummary": "Three-layer MLP QNetwork (11 -> 256 -> 256 -> 3) with Kaiming Uniform initialization and ReLU activations, paired with a hard-synced target network and optional DuelingQNetwork support.",
    "configurationBadge": "500 Ep CPU Run",
    "hyperparameters": [
      {
        "name": "hidden_dims",
        "value": "[256, 256]",
        "description": "Width of the two ReLU-activated hidden linear layers in QNetwork."
      },
      {
        "name": "learning_rate",
        "value": "0.0005",
        "description": "Adam step size balancing fast value convergence and stable TD updates."
      },
      {
        "name": "gamma",
        "value": "0.99",
        "description": "Discount factor prioritizing long-term survival and future apple chains."
      },
      {
        "name": "batch_size / buffer_capacity",
        "value": "64 / 100,000",
        "description": "Mini-batch size sampled uniformly from the cyclic replay buffer."
      },
      {
        "name": "epsilon_schedule",
        "value": "1.0 -> 0.01 (0.995)",
        "description": "Exponential per-episode exploration decay reaching 0.0816 at episode 500."
      },
      {
        "name": "target_update_freq / grad_clip",
        "value": "500 steps / 1.0",
        "description": "Hard target sync interval and L2 gradient norm threshold."
      }
    ],
    "objectiveCriterion": "Huber Loss (SmoothL1Loss) on Double DQN Bellman TD Error",
    "optimizationSolver": "Adam Optimizer with L2 Gradient Norm Clipping (1.0)",
    "trainingProcess": "Trained for 500 episodes on CPU (70.68s wall-clock) with seed 42, decaying epsilon exponentially from 1.0 to 0.0816. Model checkpoints were saved every 100 episodes and best_model.pt was selected by peak 30-episode rolling score (9.20 at episode 477), hitting 27 peak training apples at episode 457."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "100-Episode Multi-Seed Evaluation vs. Baselines",
    "description": "Greedy evaluation (epsilon = 0.0) of best_model.pt over 100 episodes split evenly across 5 unseen seeds [42, 101, 202, 303, 404], benchmarked identically against Uniform Random and Heuristic Greedy agents.",
    "metrics": [
      {
        "label": "MEAN EVAL SCORE",
        "value": "29.58",
        "benchmark": "Heuristic: 28.22"
      },
      {
        "label": "MEDIAN EVAL SCORE",
        "value": "30.0",
        "benchmark": "Heuristic: 26.5"
      },
      {
        "label": "PEAK EVAL SCORE",
        "value": "60",
        "benchmark": "Heuristic: 57"
      },
      {
        "label": "MEAN REWARD",
        "value": "+320.25",
        "benchmark": "Random: -11.42"
      },
      {
        "label": "MEAN SURVIVAL STEPS",
        "value": "451.13",
        "benchmark": "Heuristic: 432.20"
      },
      {
        "label": "SUCCESS RATE (>=15)",
        "value": "91.0%",
        "benchmark": "Target > 85%"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Dashboard Overview & Performance Snapshot",
      "badge": "Streamlit UI",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image01.webp",
      "telemetryNote": "Displays Double DQN MDP specs alongside 29.58 mean score, 60 peak score, and 91.0% success rate."
    },
    {
      "title": "Live Gameplay Demonstration Simulator",
      "badge": "Interactive Sim",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image02.webp",
      "telemetryNote": "Interactive 20x20 grid playback supporting DQN, Heuristic, and Random policies with seed controls."
    },
    {
      "title": "Training Dynamics & Loss Progression Dashboard",
      "badge": "Training View",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image03.webp",
      "telemetryNote": "Aggregates 500-episode training KPIs (27 max score, 7.93 final rolling score, 0.0816 final epsilon)."
    },
    {
      "title": "Baseline Evaluation & Comparative Benchmarks",
      "badge": "Benchmark View",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image04.webp",
      "telemetryNote": "Contrasts 100-episode statistics across DQN, Random, and Heuristic agents with boxplots and histograms."
    },
    {
      "title": "Neural State & Q-Value Inspector",
      "badge": "State Debugger",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image05.webp",
      "telemetryNote": "Interactive 11-feature toggle interface displaying real-time action Q-values (STRAIGHT Q = 14.900)."
    },
    {
      "title": "Gameplay Snapshot & Evaluation Distribution",
      "badge": "Composite Plot",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image06.webp",
      "telemetryNote": "Pairs a SnakeRenderer board frame with Q-value overlays against the 100-episode evaluation histogram."
    },
    {
      "title": "Training Reward Progression per Episode",
      "badge": "Reward Curve",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image07.webp",
      "telemetryNote": "Raw and 30-episode moving average reward rising from negative penalties to a peak of 292.65."
    },
    {
      "title": "Snake Game Score Learning Curve",
      "badge": "Score Curve",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image08.webp",
      "telemetryNote": "Tracks apples eaten across 500 episodes, annotating peak training score 27 at episode 457."
    },
    {
      "title": "Survival Episode Length (Steps)",
      "badge": "Survival Curve",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image09.webp",
      "telemetryNote": "Shows survival duration expanding from <50 steps early on to a 461-step peak at episode 457."
    },
    {
      "title": "Temporal-Difference (Huber) Training Loss",
      "badge": "Loss Curve",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image10.webp",
      "telemetryNote": "Huber TD-loss climbs smoothly from ~0.04 to ~0.49 as cumulative discounted episode returns grow."
    },
    {
      "title": "Epsilon Decay & Evaluation Boxplot Comparison",
      "badge": "Distribution Plot",
      "imageUrl": "/images/projects/strategic_game_playing_agent/image11.webp",
      "telemetryNote": "Links 0.995 exponential epsilon decay with DQN's higher evaluation median (30.0) over Heuristic (26.5)."
    }
  ],
  "mlVisualizationTelemetryNote": "Training under epsilon-greedy decay peaks at 27 apples (rolling mean 9.20), whereas deterministic greedy evaluation (epsilon = 0.0) reaches 29.58 mean apples and 60 peak apples.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Empirical evaluations show the Double DQN policy outperforms both random and greedy heuristic agents by learning perimeter-hugging turns and double-turn reversals. Q-value inspection confirms strong preference for collision-free directions aligned with food quadrants.",
    "keyDrivers": [
      {
        "factor": "Relative Danger Encoding (danger_straight/right/left)",
        "impact": "Primary safety signal preventing immediate wall and self-body collisions across orientations."
      },
      {
        "factor": "Long-Horizon Discounting (gamma = 0.99)",
        "impact": "Encourages space-preserving perimeter routing over myopic shortest-path moves."
      },
      {
        "factor": "Asymmetric Distance Shaping (+0.10 / -0.15)",
        "impact": "Eliminates back-and-forth reward looping while accelerating early food-seeking convergence."
      }
    ],
    "limitations": [
      {
        "title": "1-Cell Local Horizon",
        "description": "Because danger features only inspect adjacent cells, snakes exceeding ~40 segments can enter closed U-shaped tail pockets."
      },
      {
        "title": "Fixed Grid Geometry",
        "description": "Policy weights are calibrated to 20x20 non-wrap-around dynamics and were not evaluated on larger board dimensions."
      }
    ],
    "futureImprovements": [
      {
        "title": "CNN Multi-Channel Board Encoding",
        "target": "Global dead-end lookahead across full 20x20 spatial matrices"
      },
      {
        "title": "Prioritized Experience Replay (PER)",
        "target": "Faster convergence on rare late-game self-collision transitions"
      },
      {
        "title": "Dueling DQN Benchmarking",
        "target": "Comparative ablation using the implemented DuelingQNetwork module"
      }
    ],
    "discoveryNote": "All 15 pytest unit and integration checks passed on environment mechanics, network shapes, replay sampling, and evaluation pipelines."
  },
  "mlModelArtifacts": [
    {
      "name": "models/best_model.pt",
      "format": "PT"
    },
    {
      "name": "models/checkpoints/checkpoint_ep100.pt..ep500.pt",
      "format": "PT"
    },
    {
      "name": "metrics/training_metrics.csv",
      "format": "CSV"
    },
    {
      "name": "evaluation/evaluation_results.csv",
      "format": "CSV"
    },
    {
      "name": "evaluation/evaluation_summary.json",
      "format": "JSON"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Reinforcement Learning",
    "modelArchitecture": "Double Deep Q-Network",
    "coreFramework": "PyTorch",
    "validationMethod": "Multi-Seed Baseline Evaluation"
  }
}
```
