# VRPP Reinforcement Learning Comparison

This repository contains a reinforcement learning project focused on vessel rotation planning in maritime ports (VRPP - Vessel Rotation Planning Problem). The goal is to compare PPO (Proximal Policy Optimization) and DQN (Deep Q-Network) algorithms in a simulated port environment, with emphasis on policy behavior, adaptability, and robustness rather than raw reward maximization.

## Problem Statement

In maritime port operations, vessel rotation planning involves deciding the service priority level (Normal, Priority, or Expedited) for incoming vessels to minimize delays while managing operational costs. The environment is characterized by:

- **State Space**: Vessel characteristics (major/minor ports, seasonal factors, weight)
- **Action Space**: Three discrete actions (0: Normal, 1: Priority, 2: Expedited) with different delay multipliers (1.0, 0.85, 0.7)
- **Reward**: Negative clipped delay hours (penalizing delays)
- **Challenges**: Non-stationary environment, stochastic elements, need for balanced exploration vs. exploitation

The project evaluates how different RL algorithms adapt to this domain, focusing on policy characteristics beyond performance metrics.

## Methodology

### Environment Design
- Custom Gymnasium environments for both PPO and DQN
- State representation using normalized vessel features from operational data
- Reward shaping based on real delay statistics (clipped at 99th percentile)
- Episode length: 256 steps with random sampling from historical data

### Algorithms
- **DQN**: Deep Q-Network with experience replay, epsilon-greedy exploration
- **PPO**: Proximal Policy Optimization with actor-critic architecture
- **Baselines**: Fixed policies (always Normal, always Priority, always Expedited)

### Training Configuration
- Total timesteps: 300,000 for both algorithms
- Episode-based evaluation with action distribution tracking
- Hyperparameters tuned for stable learning in delay minimization task

## Experiments

The experimental setup includes:

1. **Baseline Evaluation**: Three fixed policies evaluated over 300 episodes
2. **RL Training**: Separate training runs for DQN and PPO
3. **Comparative Analysis**: 
   - Learning curves (smoothed episode returns/step rewards)
   - Policy behavior analysis (action distributions)
   - Determinism metrics (entropy, action concentration)
   - Early vs. late training snapshots

### Data Sources
- Preprocessed operational data (`data/`) containing vessel states and delay outcomes
- Reward statistics for normalization and clipping
- Training logs and episode-level metrics

## Results

### Performance Comparison
- PPO and DQN achieve similar final performance levels
- Baselines show clear performance hierarchies: Expedited > Priority > Normal

### Policy Characteristics
- **PPO**: Maintains mixed, stochastic policy throughout training
  - Balanced action distribution across service levels
  - Higher entropy, indicating adaptability to uncertainty
  - Suitable for non-stationary environments

- **DQN**: Converges to highly deterministic, aggressive strategy
  - Rapid shift toward Expedited service (lowest delay multiplier)
  - Low entropy, indicating strong exploitation
  - Risk of overfitting to training distribution

### Key Insights
- PPO demonstrates better robustness and adaptability
- DQN shows faster convergence but potentially brittle policies
- Determinism analysis reveals fundamental algorithmic differences in exploration-exploitation balance

## Limitations

1. **Simulation Fidelity**: Environment based on historical data; may not capture all real-world complexities
2. **Reward Design**: Simplified delay-based reward; real operations involve multiple objectives
3. **Scale**: Limited episode length and training budget; production systems require larger scale
4. **Generalization**: Trained on specific port data; transfer to other ports untested

## Future Work

1. **Multi-Objective Optimization**: Incorporate cost, resource utilization, and fairness metrics
2. **Transfer Learning**: Evaluate policy transfer across different ports
3. **Online Learning**: Adapt to changing operational conditions
4. **Ensemble Methods**: Combine multiple RL approaches for robust decision-making
5. **Real-time Deployment**: Integration with port management systems

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vrpp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Running Baselines
```bash
python -m rl_models.baseline
```

### Training DQN
```bash
python -m rl_models.train_dqn
```

### Training PPO
```bash
python -m rl_models.train_ppo
```

### Analysis
Open `outputs/combined_analysis.ipynb` in Jupyter to reproduce comparative analysis and visualizations.

## Repository Structure

```
vrpp/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/                        # Preprocessed operational data
│   ├── reward_stats.json        # Reward normalization statistics
│   ├── rl_state_base.parquet    # Training state data
│   └── *.xlsx                   # Raw and cleaned datasets
├── models/                      # Additional modeling notebooks
├── notebooks/                   # Data exploration notebooks
├── outputs/                     # Experiment results
│   ├── combined_analysis.ipynb  # Comparative analysis
│   ├── baseline/                # Baseline policy results
│   ├── dqn/                     # DQN training outputs
│   └── ppo/                     # PPO training outputs
└── rl_models/                   # Core RL implementation
    ├── baseline.py              # Baseline policy evaluation
    ├── env_dqn.py               # DQN environment
    ├── env_ppo.py               # PPO environment
    ├── train_dqn.py             # DQN training script
    └── train_ppo.py             # PPO training script
```

## Dependencies

Key libraries:
- `gymnasium`: Environment interface
- `stable-baselines3`: RL algorithms
- `torch`: Neural network backend
- `pandas`, `numpy`: Data processing
- `matplotlib`: Visualization

See `requirements.txt` for complete list.

## Contributing

Contributions welcome! Please open issues for bugs or feature requests, and submit pull requests for improvements.

## License

[Specify license if applicable]
