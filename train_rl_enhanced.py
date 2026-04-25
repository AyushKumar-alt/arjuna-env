import argparse
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from rl_env_wrapper import ArjunaTask3Wrapper

class EnhancedTrackingCallback(BaseCallback):
    """
    Enhanced callback for comprehensive logging during RL training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.history = []
        self.episode_count = 0
        self.start_time = datetime.now()

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones") is not None:
            done = self.locals["dones"][0]
            if done:
                self.episode_count += 1
                info = self.locals.get("infos", [{}])[0]

                # Extract reward
                ep_reward = info.get("overall_reward", 0.0)

                # Extract additional metrics
                final_decision = info.get("final_decision", "unknown")
                elapsed_time = (datetime.now() - self.start_time).total_seconds()

                self.history.append({
                    "timestep": self.num_timesteps,
                    "episode": self.episode_count,
                    "reward": ep_reward,
                    "decision": final_decision,
                    "elapsed_time": elapsed_time
                })

                # Progress logging every 100 episodes
                if self.episode_count % 100 == 0:
                    avg_reward = np.mean([h["reward"] for h in self.history[-100:]])
                    print(f"Episode {self.episode_count}: Avg Reward (last 100) = {avg_reward:.3f}")

        return True

def run_experiment(experiment_name: str, total_timesteps: int, difficulty_override=None,
                  learning_rate=0.001, network_arch=[64, 64], random_seed=42):
    """
    Run a single RL experiment with specified parameters.
    """
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Timesteps: {total_timesteps}, LR: {learning_rate}, Arch: {network_arch}")
    print(f"Difficulty Override: {difficulty_override}")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    set_random_seed(random_seed)

    # Initialize environment
    env = ArjunaTask3Wrapper(override_difficulty=difficulty_override)
    env = Monitor(env)

    # Configure device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build PPO Agent with enhanced architecture
    policy_kwargs = dict(
        net_arch=network_arch,
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=128,  # Larger batch for better learning
        batch_size=64,
        n_epochs=10,  # More epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Small entropy bonus
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=device,
        verbose=1
    )

    # Train with callback
    tracker = EnhancedTrackingCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=tracker,
        progress_bar=True
    )

    # Save model and results
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    model_path = f"models/ppo_{experiment_name}_{total_timesteps}ts"
    model.save(model_path)

    results_df = pd.DataFrame(tracker.history)
    results_path = f"results/training_log_{experiment_name}.csv"
    results_df.to_csv(results_path, index=False)

    # Save training summary
    summary = {
        "experiment_name": experiment_name,
        "total_timesteps": total_timesteps,
        "final_avg_reward": results_df["reward"].tail(100).mean(),
        "total_episodes": len(results_df),
        "learning_rate": learning_rate,
        "network_arch": str(network_arch),
        "device": device,
        "difficulty_override": difficulty_override
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"results/summary_{experiment_name}.csv", index=False)

    print(f"✓ Experiment '{experiment_name}' completed!")
    print(f"  Model saved to: {model_path}")
    print(f"  Results saved to: {results_path}")
    print(f"  Final Avg Reward: {results_df['reward'].tail(100).mean():.3f}")
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Enhanced RL Training for Arjuna Environment")
    parser.add_argument("--timesteps", type=int, default=5000,
                       help="Total training timesteps per experiment")
    parser.add_argument("--experiments", type=str, default="comparison",
                       help="Which experiments to run: 'all', 'curriculum', 'hard_only', 'comparison'")
    args = parser.parse_args()

    print("🤖 ARJUNA RL Training Suite")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    experiments_to_run = []

    if args.experiments == "all":
        experiments_to_run = [
            ("curriculum", None, 0.001, [64, 64]),
            ("hard_only", "hard", 0.001, [64, 64]),
            ("curriculum_large_net", None, 0.001, [128, 128]),
            ("curriculum_small_lr", None, 0.0001, [64, 64]),
        ]
    elif args.experiments == "curriculum":
        experiments_to_run = [("curriculum", None, 0.001, [64, 64])]
    elif args.experiments == "hard_only":
        experiments_to_run = [("hard_only", "hard", 0.001, [64, 64])]
    elif args.experiments == "comparison":
        experiments_to_run = [
            ("curriculum", None, 0.001, [64, 64]),
            ("hard_only", "hard", 0.001, [64, 64]),
        ]

    all_results = {}

    for exp_name, diff_override, lr, arch in experiments_to_run:
        results_df = run_experiment(
            experiment_name=exp_name,
            total_timesteps=args.timesteps,
            difficulty_override=diff_override,
            learning_rate=lr,
            network_arch=arch
        )
        all_results[exp_name] = results_df

    print(f"\n{'='*60}")
    print("🎉 ALL EXPERIMENTS COMPLETED!")
    print("Run 'python analyze_results.py' to generate comprehensive analysis and plots.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()