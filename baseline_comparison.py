import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from rl_env_wrapper import ArjunaTask3Wrapper

class QLearningAgent:
    """Simple Q-learning implementation for baseline comparison."""

    def __init__(self, state_size=1, action_size=3, learning_rate=0.01,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Discretize state space (confidence 0.0-1.0 into 20 bins)
        self.state_bins = np.linspace(0.0, 1.0, 21)
        self.q_table = np.zeros((20, action_size))

    def discretize_state(self, state):
        """Convert continuous state to discrete."""
        return np.digitize(state[0], self.state_bins) - 1

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            discrete_state = self.discretize_state(state)
            return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, done):
        """Q-learning update."""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        # Q-learning formula
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next_state])

        self.q_table[discrete_state, action] += self.learning_rate * (target - self.q_table[discrete_state, action])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QLearningCallback(BaseCallback):
    """Callback for Q-learning training."""

    def __init__(self, agent, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        self.history = []
        self.episode_count = 0
        self.start_time = datetime.now()

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            done = self.locals["dones"][0]
            if done:
                self.episode_count += 1
                info = self.locals.get("infos", [{}])[0]
                ep_reward = info.get("overall_reward", 0.0)
                final_decision = info.get("final_decision", "unknown")
                elapsed_time = (datetime.now() - self.start_time).total_seconds()

                self.history.append({
                    "timestep": self.num_timesteps,
                    "episode": self.episode_count,
                    "reward": ep_reward,
                    "epsilon": self.agent.epsilon,
                    "decision": final_decision,
                    "elapsed_time": elapsed_time
                })

                if self.episode_count % 100 == 0:
                    avg_reward = np.mean([h["reward"] for h in self.history[-100:]])
                    print(f"Q-Learning Episode {self.episode_count}: Avg Reward = {avg_reward:.3f}, Epsilon = {self.agent.epsilon:.3f}")

        return True

def run_q_learning_experiment(total_episodes=5000, difficulty_override=None, output_name="q_learning"):
    """Run Q-learning baseline experiment."""
    print(f"🧠 Running Q-Learning Baseline Experiment: {output_name}")

    env = ArjunaTask3Wrapper(override_difficulty=difficulty_override)
    env = Monitor(env)

    agent = QLearningAgent()
    callback = QLearningCallback(agent)

    episode_rewards = []

    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # Manual callback trigger
        callback.num_timesteps = episode + 1
        callback.locals = {
            "dones": [done],
            "infos": [info]
        }
        callback._on_step()

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            print(f"Episode {episode + 1}/{total_episodes}: Avg Reward (last 500) = {avg_reward:.3f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(callback.history)
    results_path = f"results/training_log_{output_name}.csv"
    results_df.to_csv(results_path, index=False)

    # Save model and results
    os.makedirs("models", exist_ok=True)
    model_path = f"models/q_learning_{output_name}.npz"
    np.savez(model_path, q_table=agent.q_table, epsilon=agent.epsilon)
    print(f"Saved Q-learning model: {model_path}")

    summary = {
        "algorithm": "Q-Learning",
        "total_episodes": total_episodes,
        "final_avg_reward": results_df["reward"].tail(500).mean(),
        "final_epsilon": results_df["epsilon"].tail(1).iloc[0],
        "difficulty_override": difficulty_override,
        "model_path": model_path
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"results/summary_{output_name}.csv", index=False)

    print("✓ Q-Learning experiment completed!")
    return results_df

def run_dqn_experiment(total_timesteps=5000, difficulty_override=None, output_name="dqn"):
    """Run DQN baseline experiment."""
    print(f"🎯 Running DQN Baseline Experiment: {output_name}")

    env = ArjunaTask3Wrapper(override_difficulty=difficulty_override)
    env = Monitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        device=device,
        verbose=1
    )

    class DQNCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.history = []
            self.episode_count = 0
            self.start_time = datetime.now()

        def _on_step(self) -> bool:
            if self.locals.get("dones") is not None:
                done = self.locals["dones"][0]
                if done:
                    self.episode_count += 1
                    info = self.locals.get("infos", [{}])[0]
                    ep_reward = info.get("overall_reward", 0.0)
                    final_decision = info.get("final_decision", "unknown")
                    elapsed_time = (datetime.now() - self.start_time).total_seconds()

                    self.history.append({
                        "timestep": self.num_timesteps,
                        "episode": self.episode_count,
                        "reward": ep_reward,
                        "decision": final_decision,
                        "elapsed_time": elapsed_time
                    })

                    if self.episode_count % 100 == 0:
                        avg_reward = np.mean([h["reward"] for h in self.history[-100:]])
                        print(f"DQN Episode {self.episode_count}: Avg Reward = {avg_reward:.3f}")

            return True

    callback = DQNCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save results
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    model_path = f"models/{output_name}"
    model_path = f"models/{output_name}"
    model.save(model_path)
    results_df = pd.DataFrame(callback.history)
    results_df.to_csv(f"results/training_log_{output_name}.csv", index=False)

    summary = {
        "algorithm": "DQN",
        "total_timesteps": total_timesteps,
        "final_avg_reward": results_df["reward"].tail(100).mean(),
        "total_episodes": len(results_df),
        "difficulty_override": difficulty_override,
        "device": device,
        "model_path": model_path
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"results/summary_{output_name}.csv", index=False)

    print("✓ DQN experiment completed!")
    return results_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline RL algorithm comparisons")
    parser.add_argument("--algorithms", nargs="+", default=["q_learning", "dqn"],
                       help="Algorithms to run: q_learning, dqn")
    parser.add_argument("--episodes", type=int, default=3000,
                       help="Episodes for Q-learning")
    parser.add_argument("--timesteps", type=int, default=5000,
                       help="Timesteps for DQN")
    args = parser.parse_args()

    print("🧪 ARJUNA BASELINE ALGORITHMS COMPARISON")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if "q_learning" in args.algorithms:
        run_q_learning_experiment(total_episodes=args.episodes)

    if "dqn" in args.algorithms:
        run_dqn_experiment(total_timesteps=args.timesteps)

    print("\n🎉 Baseline experiments completed!")
    print("Run 'python analyze_results.py --report' for comprehensive analysis.")

if __name__ == "__main__":
    main()