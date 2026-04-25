
# ARJUNA System Architecture

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI        │    │   RL Agent      │
│   Dashboard     │◄──►│   Environment    │◄──►│   (PPO/DQN)     │
│   (app.py)      │    │   (server/)      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Curriculum    │    │   Synthetic      │    │   Training      │
│   Manager       │    │   Data           │    │   Pipeline      │
│   (curriculum.py)│    │   (synthetic_data.py)│    │   (train_*.py) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Component Details

### Environment Server (server/)
- **arjuna_environment.py:** Main MDP implementation
- **synthetic_data.py:** Offline episode bundles (14 scenarios)
- **tasks.py:** Grading logic with dense rewards
- **curriculum.py:** Auto-curriculum difficulty adjustment

### RL Training Components
- **rl_env_wrapper.py:** Gymnasium wrapper for Task 3 focus
- **train_rl_enhanced.py:** PPO training with GPU support
- **baseline_comparison.py:** Q-learning and DQN baselines

### Analysis & Visualization
- **analyze_results.py:** Comprehensive result analysis
- **app.py:** Streamlit dashboard for monitoring
- **plot_rl_results.py:** Training curve generation

## Data Flow

### Training Loop
1. **Reset:** Environment selects bundle based on curriculum
2. **Task 1:** Agent identifies primary object
3. **Task 2:** Agent ranks objects by priority
4. **Task 3:** Agent makes confidence-based decision
5. **Reward:** Dense sequence alignment scoring
6. **Curriculum Update:** Adjust difficulty based on performance

### Curriculum Integration
- Tracks agent performance over sliding windows
- Promotes on sustained good performance
- Demotes on consistent poor performance
- Prevents policy oscillation through window clearing

## Technical Specifications

### Dependencies
- **PyTorch:** Deep learning framework
- **Stable Baselines3:** RL algorithm implementations
- **FastAPI:** Web server framework
- **Streamlit:** Dashboard framework
- **Gymnasium:** Environment interface

### Hardware Requirements
- **GPU:** CUDA-compatible for accelerated training
- **RAM:** 8GB+ for parallel environment execution
- **Storage:** 10GB+ for models and training data

### Performance Characteristics
- **Episode Length:** 3 steps (fixed)
- **Training Speed:** ~100 episodes/minute on GPU
- **Convergence:** 1000-5000 episodes depending on difficulty
- **Memory Usage:** ~2GB during training
