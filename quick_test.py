#!/usr/bin/env python3
"""
Quick test of ARJUNA environment and PPO training.
Minimal version without pandas dependency for initial testing.
"""

import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from rl_env_wrapper import ArjunaTask3Wrapper

class SimpleCallback(BaseCallback):
    """Simple callback for testing."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            done = self.locals["dones"][0]
            if done:
                self.episode_count += 1
                info = self.locals.get("infos", [{}])[0]
                ep_reward = info.get("overall_reward", 0.0)
                self.episode_rewards.append(ep_reward)

                if self.episode_count % 50 == 0:
                    avg_reward = np.mean(self.episode_rewards[-50:])
                    print(f"Episode {self.episode_count}: Avg Reward = {avg_reward:.3f}")

        return True

def test_environment():
    """Test basic environment functionality."""
    print("🧪 Testing ARJUNA Environment...")

    env = ArjunaTask3Wrapper()
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    print(f"Info: {info}")

    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}")

        if done:
            break

    print("✅ Environment test passed!")

def test_ppo_training():
    """Test PPO training."""
    print("\n🤖 Testing PPO Training...")

    env = ArjunaTask3Wrapper()
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        n_steps=64,
        batch_size=32,
        verbose=1
    )

    callback = SimpleCallback()
    model.learn(total_timesteps=500, callback=callback)

    print("✅ PPO training test passed!")
    print(f"Total episodes: {callback.episode_count}")
    print(f"Final avg reward: {np.mean(callback.episode_rewards[-50:]):.3f}")

def main():
    print("🚀 ARJUNA RL QUICK TEST")
    print("=" * 40)

    try:
        test_environment()
        test_ppo_training()

        print("\n🎉 ALL TESTS PASSED!")
        print("The ARJUNA environment and PPO training are working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()