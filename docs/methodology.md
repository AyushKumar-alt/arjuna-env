
# ARJUNA RL Methodology

## Algorithm Selection: Why PPO?

We selected Proximal Policy Optimization (PPO) for the following reasons:

### Theoretical Justification
1. **Sample Efficiency:** PPO achieves better sample efficiency than Q-learning
2. **Stability:** Clipped surrogate objective prevents destructive policy updates
3. **Continuous Action Handling:** Although our actions are discrete, PPO generalizes well
4. **Off-Policy Learning:** Can learn from experiences collected with older policies

### Problem Fit
1. **Episodic Nature:** ARJUNA episodes are short (3 steps), suitable for PPO's trajectory optimization
2. **Reward Scale:** Dense rewards [0,1] work well with PPO's advantage estimation
3. **State Space:** Simple state representation (confidence scalar) minimizes complexity

### Compared to Alternatives
- **Q-Learning:** Tabular methods struggle with continuous confidence values
- **DQN:** Requires large replay buffers; overkill for simple state space
- **Policy Gradient:** Higher variance than PPO's clipped objective

## Implementation Details

### Network Architecture
```
Input (1) → Hidden 1 (64) → Hidden 2 (64) → Output (3)
Activation: ReLU
Output: Softmax for action probabilities
```

### Hyperparameters
- Learning Rate: 0.001 (Adam optimizer)
- Batch Size: 64
- Mini-batches: 4 per update
- Epochs per update: 10
- Clip Range: 0.2
- Value Function Coefficient: 0.5
- Entropy Coefficient: 0.01

### Training Strategy
1. **Environment Reset:** Initialize episode with random bundle
2. **Task Progression:** Auto-advance through tasks 1→2→3
3. **Reward Calculation:** Dense sequence alignment rewards
4. **Curriculum Integration:** Difficulty adjusts based on performance
5. **Early Stopping:** Monitor convergence on validation episodes

## Exploration Strategy

### ε-Greedy Exploration (PPO Default)
- Initial exploration through stochastic policy sampling
- Entropy bonus (0.01) encourages exploration
- Annealing through policy improvement

### Curriculum as Exploration
- Auto-curriculum provides structured exploration of difficulty levels
- Prevents agent from getting stuck in easy scenarios
- Promotes robust policy learning

## Reward Engineering

### Dense Rewards Design
- **Sequence Alignment:** Levenshtein distance for partial credit
- **Semantic Grouping:** Category-level rewards for object identification
- **Reasoning Quality:** Bonus for confidence-aware decisions

### Normalization
- All rewards clamped to [0.01, 0.99] to ensure gradient flow
- Overall episode reward as mean of step rewards

## Evaluation Metrics

### Performance Metrics
- **Episode Reward:** Mean reward across 3 tasks
- **Convergence:** Reward stability over sliding windows
- **Decision Accuracy:** Correct action selection rates
- **Curriculum Progression:** Speed of difficulty advancement

### Baselines for Comparison
- **Random Policy:** Random action selection
- **Heuristic Policy:** Rule-based decisions
- **Q-Learning:** Tabular RL baseline
- **DQN:** Deep RL baseline
