import argparse
import os
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from rl_env_wrapper import ArjunaTask3Wrapper

class TrackingCallback(BaseCallback):
    """
    Custom callback for logging metrics during training specifically for the course project.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.history = []
        
    def _on_step(self) -> bool:
        # Our environment is a 1-step episode. So every step is an episode end.
        done = self.locals.get("dones")[0]
        if done:
            info = self.locals.get("infos")[0]
            # Try to grab the reward either from the monitor or directly from info
            ep_reward = info.get("overall_reward")
            if ep_reward is None and "episode" in info:
                ep_reward = info["episode"]["r"]
                
            self.history.append({
                "timestep": self.num_timesteps,
                "reward": ep_reward,
                "decision": info.get("final_decision", "unknown")
            })
        return True

def run_experiment(mode: str, total_timesteps: int):
    print(f"--- Starting RL Training: [{mode}] Mode ---")
    
    # 1. Initialize our gym wrapper
    override = "hard" if mode == "hard_only" else None
    env = ArjunaTask3Wrapper(override_difficulty=override)
    
    # Monitor to keep track of basic stats
    env = Monitor(env)

    # 2. Build the PPO Agent
    # Since it's a simple 1D state space, a tiny network learns very fast.
    policy_kwargs = dict(net_arch=[16, 16])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                learning_rate=0.01, n_steps=64, batch_size=64, 
                verbose=0)

    # 3. Train
    tracker = TrackingCallback()
    model.learn(total_timesteps=total_timesteps, callback=tracker, progress_bar=True)
    
    # 4. Save results to CSV
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(tracker.history)
    df.to_csv(f"results/training_log_{mode}.csv", index=False)
    print(f"Training complete. Logs saved to results/training_log_{mode}.csv\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent on Arjuna Tasks")
    parser.add_argument("--timesteps", type=int, default=1500, help="Total training steps")
    args = parser.parse_args()

    # Run the control group (Hard Only - No Curriculum)
    run_experiment(mode="hard_only", total_timesteps=args.timesteps)
    
    # Run the experimental group (Auto-Curriculum Activated)
    run_experiment(mode="curriculum", total_timesteps=args.timesteps)
    
    print("Both experiments complete! You can now run `python plot_rl_results.py` to compile the learning curves.")
