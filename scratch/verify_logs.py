import sys
from unittest.mock import MagicMock
import os

# Mock the OpenAI and ArjunaEnv classes before importing inference
sys.modules['openai'] = MagicMock()
sys.modules['client'] = MagicMock()
sys.modules['models'] = MagicMock()

import inference

# Set up mocks for environment
mock_env = MagicMock()
mock_obs = MagicMock()
mock_obs.episode_id = "test-ep-123"
mock_obs.scene_id = "test-scene"
mock_obs.task_type = 1
mock_obs.observation_text = "Bundle: Warehouse\nTest scene"
mock_obs.overall_reward = 1.0

mock_reset = MagicMock()
mock_reset.observation = mock_obs
mock_env.reset.return_value = mock_reset

mock_step_out = MagicMock()
mock_step_out.reward = 1.0
mock_step_out.done = False
mock_env.step.return_value = mock_step_out

# Second step
def side_effect(action):
    if hasattr(action, 'task1_label'):
        mock_step_out.done = False
        mock_obs.task_type = 2
        return mock_step_out
    if hasattr(action, 'ranked_objects'):
        mock_step_out.done = False
        mock_obs.task_type = 3
        return mock_step_out
    if hasattr(action, 'decision'):
        mock_step_out.done = True
        return mock_step_out
    return mock_step_out

mock_env.step.side_effect = side_effect

# Run 1 episode
inference.MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
inference.HF_TOKEN = "test_token"

try:
    with MagicMock() as env_context:
        # Simulate run_episode
        inference.ArjunaEnv = MagicMock(return_value=mock_env)
        mock_env.sync.return_value.__enter__.return_value = mock_env
        
        # Override _chat to be fast
        inference._chat = MagicMock(side_effect=["person", '["truck"]', '{"decision": "discard"}'])
        
        # We need to mock the per_task and other globals if needed, 
        # but let's just trigger run_episode directly.
        inference.per_task = {1:[], 2:[], 3:[]}
        
        inference.run_episode(mock_env, MagicMock(), seed=42, ep_idx=1)
except Exception as e:
    print(f"Error in test: {e}")
