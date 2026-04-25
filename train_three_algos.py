import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from train_rl_enhanced import run_experiment
from baseline_comparison import run_dqn_experiment, run_q_learning_experiment

sns.set_theme(style="whitegrid")


def smooth(series, weight=0.85):
    smoothed = []
    last = series.iloc[0]
    for v in series:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_algorithm_results(algorithm_name, log_files, labels, output_file):
    plt.figure(figsize=(10, 6))

    for log_file, label in zip(log_files, labels):
        if not os.path.exists(log_file):
            print(f"WARNING: missing log file {log_file}")
            continue
        df = pd.read_csv(log_file)
        if "episode" not in df.columns:
            df["episode"] = range(1, len(df) + 1)
        df["smoothed_reward"] = smooth(df["reward"])
        plt.plot(df["episode"], df["smoothed_reward"], label=label, linewidth=2)

    plt.title(f"{algorithm_name} Performance: Curriculum vs Hard-Only")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_decision_distribution(log_file, title, output_file):
    if not os.path.exists(log_file):
        print(f"WARNING: missing decision log file {log_file}")
        return
    df = pd.read_csv(log_file)
    if "decision" not in df.columns:
        print(f"WARNING: no decision column in {log_file}")
        return
    counts = df["decision"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="#4C72B0")
    plt.title(title)
    plt.xlabel("Decision")
    plt.ylabel("Frequency")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Saved decision distribution plot: {output_file}")
    plt.close()


def save_summary_table(summary_files, output_file="results/summary_all_algorithms.csv"):
    rows = []
    for sf in summary_files:
        if os.path.exists(sf):
            df = pd.read_csv(sf)
            rows.append(df)
    if rows:
        combined = pd.concat(rows, ignore_index=True, sort=False)
        combined.to_csv(output_file, index=False)
        print(f"Saved combined summary table: {output_file}")
    else:
        print("No summary files found to combine.")


def train_ppo_cases(timesteps=10000):
    print("\n=== TRAINING PPO: Curriculum ===")
    run_experiment(experiment_name="curriculum", total_timesteps=timesteps, difficulty_override=None)
    print("\n=== TRAINING PPO: Hard Only ===")
    run_experiment(experiment_name="hard_only", total_timesteps=timesteps, difficulty_override="hard")
    plot_algorithm_results(
        algorithm_name="PPO",
        log_files=["results/training_log_curriculum.csv", "results/training_log_hard_only.csv"],
        labels=["PPO (Curriculum)", "PPO (Hard Only)"],
        output_file="results/plot_ppo_curriculum_vs_hard_only.png"
    )
    plot_decision_distribution(
        "results/training_log_curriculum.csv",
        "PPO Curriculum Decision Distribution",
        "results/decision_distribution_ppo_curriculum.png"
    )
    plot_decision_distribution(
        "results/training_log_hard_only.csv",
        "PPO Hard-Only Decision Distribution",
        "results/decision_distribution_ppo_hard_only.png"
    )


def train_dqn_cases(timesteps=10000):
    print("\n=== TRAINING DQN: Curriculum ===")
    run_dqn_experiment(total_timesteps=timesteps, difficulty_override=None, output_name="dqn_curriculum")
    print("\n=== TRAINING DQN: Hard Only ===")
    run_dqn_experiment(total_timesteps=timesteps, difficulty_override="hard", output_name="dqn_hard_only")
    plot_algorithm_results(
        algorithm_name="DQN",
        log_files=["results/training_log_dqn_curriculum.csv", "results/training_log_dqn_hard_only.csv"],
        labels=["DQN (Curriculum)", "DQN (Hard Only)"],
        output_file="results/plot_dqn_curriculum_vs_hard_only.png"
    )
    plot_decision_distribution(
        "results/training_log_dqn_curriculum.csv",
        "DQN Curriculum Decision Distribution",
        "results/decision_distribution_dqn_curriculum.png"
    )
    plot_decision_distribution(
        "results/training_log_dqn_hard_only.csv",
        "DQN Hard-Only Decision Distribution",
        "results/decision_distribution_dqn_hard_only.png"
    )


def train_q_learning_cases(episodes=10000):
    print("\n=== TRAINING Q-LEARNING: Curriculum ===")
    run_q_learning_experiment(total_episodes=episodes, difficulty_override=None, output_name="q_learning_curriculum")
    print("\n=== TRAINING Q-LEARNING: Hard Only ===")
    run_q_learning_experiment(total_episodes=episodes, difficulty_override="hard", output_name="q_learning_hard_only")
    plot_algorithm_results(
        algorithm_name="Q-Learning",
        log_files=["results/training_log_q_learning_curriculum.csv", "results/training_log_q_learning_hard_only.csv"],
        labels=["Q-Learning (Curriculum)", "Q-Learning (Hard Only)"],
        output_file="results/plot_q_learning_curriculum_vs_hard_only.png"
    )
    plot_decision_distribution(
        "results/training_log_q_learning_curriculum.csv",
        "Q-Learning Curriculum Decision Distribution",
        "results/decision_distribution_q_learning_curriculum.png"
    )
    plot_decision_distribution(
        "results/training_log_q_learning_hard_only.csv",
        "Q-Learning Hard-Only Decision Distribution",
        "results/decision_distribution_q_learning_hard_only.png"
    )


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_ppo_cases(timesteps=10000)
    train_dqn_cases(timesteps=10000)
    train_q_learning_cases(episodes=10000)

    save_summary_table([
        "results/summary_curriculum.csv",
        "results/summary_hard_only.csv",
        "results/summary_dqn_curriculum.csv",
        "results/summary_dqn_hard_only.csv",
        "results/summary_q_learning_curriculum.csv",
        "results/summary_q_learning_hard_only.csv"
    ], output_file="results/summary_all_algorithms.csv")

    print("\n=== ALL ALGORITHMS TRAINED AND PLOTTED ===")
    print("Plot files saved in results/ and model files saved in models/")


if __name__ == "__main__":
    main()
