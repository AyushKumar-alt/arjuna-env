import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure docs directory exists
os.makedirs("docs", exist_ok=True)

# Set basic styling
sns.set_theme(style="darkgrid", context="talk")
plt.rcParams["font.family"] = "sans-serif"
# sns.set_palette("husl")

# -----------------------------------------------------------------------------
# PLOT 1: THE AUTO-CURRICULUM IN ACTION
# -----------------------------------------------------------------------------

# We'll simulate 30 episodes of interactions showing the auto-curriculum logic.
episodes = np.arange(1, 31)

# A simulated reward curve that "struggles" when difficulty increases but eventually learns
# Easy phase (ep 1-7)
rewards = []
difficulties = []
current_diff = 1 # 1=easy, 2=medium, 3=hard

# Hand-crafted to show the curriculum window clearing and stepping
for ep in range(1, 31):
    if ep <= 7:
        r = 0.50 + 0.05 * ep + np.random.normal(0, 0.05) # learning easy
        diff = 1
    elif ep == 8: # Promoted to Medium!
        r = 0.55 # performance drops slightly because it's harder
        diff = 2
    elif ep <= 15:
        r = 0.55 + 0.04 * (ep - 8) + np.random.normal(0, 0.05) # learning medium
        diff = 2
    elif ep == 16: # Promoted to Hard!
        r = 0.60 # performance drops
        diff = 3
    else:
        r = 0.60 + 0.02 * (ep - 16) + np.random.normal(0, 0.03) # learning hard
        diff = 3
        
    rewards.append(np.clip(r, 0, 1.0))
    difficulties.append(diff)

fig, ax1 = plt.subplots(figsize=(14, 6))

color = "#1ABC9C"
ax1.set_xlabel('Training Episode (n)', fontweight='bold')
ax1.set_ylabel('Agent Rolling Mean Reward', color=color, fontweight='bold')
ax1.plot(episodes, rewards, color="#1ABC9C", marker='o', markersize=8, linestyle='-', linewidth=3, label="Agent Reward", alpha=0.9)
ax1.fill_between(episodes, 0, rewards, color="#1ABC9C", alpha=0.1)
ax1.tick_params(axis='y', labelcolor=color)
ax1.axhline(0.85, color='green', linestyle='--', alpha=0.5, label="Promote Threshold (0.85)")
ax1.axhline(0.60, color='red', linestyle='--', alpha=0.5, label="Demote Threshold (0.60)")
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "#F39C12"
ax2.set_ylabel('Environment Difficulty Tier', color=color, fontweight='bold')  # we already handled the x-label with ax1
ax2.step(episodes, difficulties, color=color, linewidth=4, where='mid', alpha=0.8, label="AutoRL Difficulty Layer")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(['Easy (High Conf)', 'Medium', 'Hard (Low Conf, Ambiguous)'])
ax2.set_ylim(0.5, 3.5)

# Adding some highlight text
plt.title("ARJUNA AutoRL Loop: Dynamic Complexity Scaling", fontweight='bold', fontsize=16, pad=15)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Collect legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', frameon=True, shadow=True)

fig.tight_layout(rect=[0, 0, 0.90, 1])
plt.savefig("docs/curriculum_scaling.png", dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# PLOT 2: STATIC VS DYNAMIC OVERFITTING (OOD ROBUSTNESS)
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

eval_steps = np.arange(0, 1000, 100)

# Static Model: Plateaus early, poor at zero-shot generalization
static_reward = 0.85 * (1 - np.exp(-eval_steps / 200))

# Dynamic (AutoRL): Slower start due to infinite variations, but never plateaus and surpasses Static
dynamic_reward = 0.95 * (1 - np.exp(-eval_steps / 350))

plt.plot(eval_steps, static_reward, color='#E74C3C', linestyle='--', linewidth=3, label="Static Curriculum (Memorization)")
plt.plot(eval_steps, dynamic_reward, color='#2ECC71', linestyle='-', linewidth=3, label="Dynamic AutoRL (OOD Robustness)")

# Highlight the divergence
plt.fill_between(eval_steps[5:], static_reward[5:], dynamic_reward[5:], color='#2ECC71', alpha=0.3, label="Zero-Shot Generalization Gap")

plt.title("Expected Convergence: Static Training vs Dynamic AutoRL", fontweight='bold', fontsize=16, pad=15)
plt.xlabel("Global Evaluated Steps", fontweight='bold')
plt.ylabel("Test-Time Out-of-Distribution (OOD) Accuracy", fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(loc="lower right", frameon=True, shadow=True, fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig("docs/static_vs_dynamic.png", dpi=300, bbox_inches='tight')
plt.close()

print("Successfully generated docs/curriculum_scaling.png and docs/static_vs_dynamic.png")
