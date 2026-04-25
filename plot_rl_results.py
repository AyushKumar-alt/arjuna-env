import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth(scalars, weight=0.85):
    """
    Exponential moving average smoothing.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_experiments():
    # Set seaborn styling for an academic looking report
    sns.set_theme(style="whitegrid")
    
    file_hard = "results/training_log_hard_only.csv"
    file_curr = "results/training_log_curriculum.csv"
    
    if not (os.path.exists(file_hard) and os.path.exists(file_curr)):
        print("Training logs not found! Please run `python train_rl.py` first.")
        return
        
    df_hard = pd.read_csv(file_hard)
    df_curr = pd.read_csv(file_curr)
    
    # Smooth the rewards
    df_hard["smoothed_reward"] = smooth(df_hard["reward"].values)
    df_curr["smoothed_reward"] = smooth(df_curr["reward"].values)
    
    plt.figure(figsize=(10, 6))
    
    # Plot smoothed lines
    plt.plot(df_curr["timestep"], df_curr["smoothed_reward"], 
             label="PPO with Auto-Curriculum", color="#1ABC9C", linewidth=2.5)
    plt.plot(df_hard["timestep"], df_hard["smoothed_reward"], 
             label="PPO (Hard-Only / No Curriculum)", color="#E74C3C", linewidth=2.5)
             
    # Plot semi-transparent raw scores using alpha
    plt.plot(df_curr["timestep"], df_curr["reward"], color="#1ABC9C", alpha=0.15)
    plt.plot(df_hard["timestep"], df_hard["reward"], color="#E74C3C", alpha=0.15)
    
    plt.title("RL Agent Performance: Task 3 Threshold Decision", fontsize=16, pad=15)
    plt.xlabel("Training Episodes (Timesteps)", fontsize=13)
    plt.ylabel("Episode Reward", fontsize=13)
    plt.ylim(0, 1.05)
    
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    out_file = "results/rl_course_project_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot successfully saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    plot_experiments()
