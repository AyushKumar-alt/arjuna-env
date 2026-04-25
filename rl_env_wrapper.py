import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models import ArjunaAction
from server.arjuna_environment import ArjunaEnvironment, SESSIONS
from server.curriculum import set_curriculum_override, get_curriculum_stats

class ArjunaTask3Wrapper(gym.Env):
    """
    A Gymnasium wrapper around ArjunaEnvironment that focuses the RL agent
    solely on Task 3 (the low-confidence threshold decision).
    
    It auto-plays through Task 1 and Task 2 using ground-truth data so
    the RL agent can rapidly learn the thresholding policy.
    """
    
    def __init__(self, override_difficulty=None):
        super().__init__()
        
        self.arjuna = ArjunaEnvironment()
        
        # Observation is a single float representing the bounding box "confidence" (0.0 to 1.0)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Actions: 0=discard, 1=request_rescan, 2=log_and_continue
        self.action_space = spaces.Discrete(3)
        
        self.episode_id = None
        
        # If set to "hard", locks the environment to "hard" mode, bypassing the curriculum.
        if override_difficulty:
            set_curriculum_override(override_difficulty)
        else:
            set_curriculum_override(None) # Allow dynamic curriculum to run

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Reset underlying environment
        obs = self.arjuna.reset(seed=seed)
        self.episode_id = obs.episode_id
        
        # Fast-forward through Task 1 and Task 2 using Ground Truth
        sess = SESSIONS.get(self.episode_id)
        if not sess:
            raise RuntimeError("Underlying session not found in ArjunaEnvironment")
            
        bundle = sess.bundle
        
        # Step 1: Identify optimally
        action1 = ArjunaAction(
            task1_label=bundle.task1.expected_label, 
            metadata={"episode_id": self.episode_id}
        )
        self.arjuna.step(action1)
        
        # Step 2: Triage optimally
        action2 = ArjunaAction(
            ranked_objects=list(bundle.task2.expected_priority),
            metadata={"episode_id": self.episode_id}
        )
        self.arjuna.step(action2)
        
        # Now we are at Task 3. Extract the confidence from the scene payload.
        confidence = bundle.task3.primary_detection.confidence
        state = np.array([confidence], dtype=np.float32)
        
        info = {
            "difficulty": get_curriculum_stats()["current_difficulty"],
            "bundle": bundle.name
        }
        return state, info

    def step(self, action):
        # Map Discrete action int to string expected by Arjuna
        mapping = {
            0: "discard",
            1: "request_rescan",
            2: "log_and_continue"
        }
        decision = mapping[int(action)]
        
        # Submit the final task
        action3 = ArjunaAction(
            decision=decision, 
            reasoning="RL Gym Agent Action",
            metadata={"episode_id": self.episode_id}
        )
        obs = self.arjuna.step(action3)
        
        # Use the overall sequence reward as the RL reward (between 0 and 1.0)
        # We clamp it slightly just to be safe.
        reward = obs.overall_reward if obs.overall_reward is not None else obs.reward
        reward = max(0.0, min(1.0, float(reward)))
        
        done = obs.done or True # It's a terminal step
        truncated = False
        
        # Agent dies/ends immediately after Task 3
        next_state = np.array([0.0], dtype=np.float32)
        info = {
            "final_decision": decision,
            "overall_reward": float(reward)
        }
        
        return next_state, float(reward), done, truncated, info
