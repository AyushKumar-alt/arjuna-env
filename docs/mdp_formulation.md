
# ARJUNA Environment: Formal MDP Formulation

## Markov Decision Process (MDP) Definition

The ARJUNA perception environment is formulated as a Markov Decision Process with the following components:

### State Space (S)
The state space is defined by the current task context and scene information:

**Task 1 (Object Identification):**
- State: `s = (scene_description, detections_list, task_type=1)`
- Where `detections_list` contains YOLO-style detections with labels and confidences

**Task 2 (Multi-Object Triage):**
- State: `s = (scene_description, detections_list, task_type=2)`
- Agent must rank objects by importance using confidence + priority rules

**Task 3 (Low-Confidence Decision):**
- State: `s = (confidence_value, task_type=3)` where `confidence_value ∈ [0.0, 1.0]`
- Simplified state representation for RL training

**Formal Definition:**
```
S = {s₁, s₂, s₃} where:
- s₁: Task 1 states (text + detections)
- s₂: Task 2 states (text + detections)
- s₃: Task 3 states (confidence scalar)
```

### Action Space (A)
Actions are task-specific and discrete:

**Task 1:** `A₁ = {object_labels}` (e.g., "person", "car", "truck")
**Task 2:** `A₂ = {ranked_lists}` (permutations of detected objects)
**Task 3:** `A₃ = {"discard", "request_rescan", "log_and_continue"}`

**Formal Definition:**
```
A(s) = {
    A₁(s) if task_type(s) = 1
    A₂(s) if task_type(s) = 2
    A₃(s) if task_type(s) = 3
}
```

### Transition Function (T)
The environment follows deterministic transitions through the 3-step episode:

**T(s, a, s') = 1** for the following transitions:
- Task 1 → Task 2: After any valid object identification action
- Task 2 → Task 3: After any valid ranking action
- Task 3 → Terminal: After any decision action

**Episode Structure:**
```
s₀ → a₀ → r₀ → s₁ → a₁ → r₁ → s₂ → a₂ → r₂ → TERMINAL
```

### Reward Function (R)
Rewards are dense and sequence-aligned, designed for better RL convergence:

**Task 1 (Object ID):** `R₁(s, a) = 1.0` if exact match, `0.7` if same category, `0.2` if wrong category
**Task 2 (Triage):** `R₂(s, a) = sequence_similarity(a, ground_truth)` using Levenshtein distance
**Task 3 (Decision):** `R₃(s, a) = 1.0` if correct band decision + good reasoning

**Overall Episode Reward:** `R_episode = mean(R₁, R₂, R₃)`

### Discount Factor (γ)
γ = 0.99 (standard for episodic tasks)

### Policy (π)
The agent learns a policy π(a|s) that maps states to actions, optimized to maximize expected cumulative reward.

## Why RL Over Supervised Learning?

1. **Sequential Decision Making:** Tasks must be performed in order, with dependencies
2. **Exploration vs Exploitation:** Agent must learn when to trust low-confidence detections
3. **Generalization:** Environment includes OOD scenarios requiring adaptive behavior
4. **Credit Assignment:** Dense rewards help learn from partial successes in complex sequences

## Curriculum Learning Integration

The MDP includes a meta-level curriculum that adjusts scenario difficulty:
- **Easy:** High-confidence, clear scenarios
- **Medium:** Moderate difficulty with triage challenges
- **Hard:** Low-confidence edge cases requiring robust policies

Curriculum transitions based on sliding window performance metrics.
