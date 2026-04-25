import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RLResultsAnalyzer:
    """Comprehensive analyzer for RL training results."""

    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_experiment_data(self, experiment_name):
        """Load training data for a specific experiment."""
        csv_path = self.results_dir / f"training_log_{experiment_name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None

    def load_summary_data(self, experiment_name):
        """Load summary data for a specific experiment."""
        csv_path = self.results_dir / f"summary_{experiment_name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path).iloc[0].to_dict()
        return None

    def plot_learning_curves(self, experiments, window_size=100):
        """Plot learning curves with moving averages."""
        plt.figure(figsize=(12, 8))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, exp_name in enumerate(experiments):
            df = self.load_experiment_data(exp_name)
            if df is None:
                continue

            # Calculate moving average
            rewards = df['reward'].rolling(window=window_size, min_periods=1).mean()

            plt.plot(df['episode'], rewards,
                    label=f'{exp_name.replace("_", " ").title()}',
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8)

            # Add final performance annotation
            final_reward = rewards.iloc[-1]
            plt.annotate(f"{final_reward:.3f}",
                        xy=(df['episode'].iloc[-1], final_reward),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, color=colors[i % len(colors)])

        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward (100-episode window)', fontsize=12)
        plt.title('ARJUNA RL Training: Learning Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.results_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_curriculum_comparison(self):
        """Compare curriculum vs hard-only performance."""
        experiments = ['curriculum', 'hard_only']
        self.plot_learning_curves(experiments)

    def analyze_decision_patterns(self, experiment_name):
        """Analyze the agent's decision-making patterns."""
        df = self.load_experiment_data(experiment_name)
        if df is None:
            return None

        # Decision distribution
        decision_counts = df['decision'].value_counts()

        plt.figure(figsize=(10, 6))
        decision_counts.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title(f'Decision Distribution: {experiment_name.replace("_", " ").title()}')
        plt.xlabel('Decision')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'decision_distribution_{experiment_name}.png', dpi=300)
        plt.show()

        return decision_counts

    def calculate_performance_metrics(self, experiment_name):
        """Calculate comprehensive performance metrics."""
        df = self.load_experiment_data(experiment_name)
        if df is None:
            return None

        metrics = {}

        # Basic statistics
        metrics['total_episodes'] = len(df)
        metrics['mean_reward'] = df['reward'].mean()
        metrics['std_reward'] = df['reward'].std()
        metrics['max_reward'] = df['reward'].max()
        metrics['min_reward'] = df['reward'].min()

        # Learning progress (first vs last 100 episodes)
        if len(df) >= 200:
            first_100 = df['reward'].head(100).mean()
            last_100 = df['reward'].tail(100).mean()
            metrics['improvement'] = last_100 - first_100
            metrics['improvement_pct'] = ((last_100 - first_100) / first_100) * 100

        # Convergence analysis (check if reward stabilizes)
        if len(df) >= 300:
            recent_100 = df['reward'].tail(100)
            earlier_100 = df['reward'].iloc[-200:-100]
            metrics['converged'] = abs(recent_100.mean() - earlier_100.mean()) < 0.05

        return metrics

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        experiments = ['curriculum', 'hard_only', 'curriculum_large_net', 'curriculum_small_lr']

        print("🔬 ARJUNA RL EXPERIMENT ANALYSIS REPORT")
        print("=" * 60)

        # Load and analyze each experiment
        results_summary = {}

        for exp in experiments:
            df = self.load_experiment_data(exp)
            summary = self.load_summary_data(exp)

            if df is not None:
                metrics = self.calculate_performance_metrics(exp)
                results_summary[exp] = {
                    'data': df,
                    'summary': summary,
                    'metrics': metrics
                }

                print(f"\n📊 Experiment: {exp.replace('_', ' ').upper()}")
                print("-" * 40)
                if metrics:
                    print(f"Total Episodes: {metrics['total_episodes']}")
                    print(f"Mean Reward: {metrics['mean_reward']:.3f}")
                    print(f"Std Reward: {metrics['std_reward']:.3f}")
                    if 'improvement' in metrics:
                        print(f"Improvement (last 100): {metrics['improvement']:.3f}")
                        print(f"Improvement %: {metrics['improvement_pct']:.1f}%")
                    if 'converged' in metrics:
                        print(f"Converged: {'Yes' if metrics['converged'] else 'No'}")

        # Generate comparison plots
        available_experiments = [exp for exp in experiments if self.load_experiment_data(exp) is not None]

        if len(available_experiments) >= 2:
            print("\n📈 Generating comparison plots...")
            self.plot_learning_curves(available_experiments)

            # Decision pattern analysis
            for exp in available_experiments:
                self.analyze_decision_patterns(exp)

        # Statistical comparison between curriculum and hard_only
        if 'curriculum' in results_summary and 'hard_only' in results_summary:
            curr_rewards = results_summary['curriculum']['data']['reward']
            hard_rewards = results_summary['hard_only']['data']['reward']

            # Perform t-test if we have enough data
            if len(curr_rewards) > 30 and len(hard_rewards) > 30:
                t_stat, p_value = stats.ttest_ind(curr_rewards.tail(200), hard_rewards.tail(200))
                print("\n🧪 Statistical Analysis:")
                print(f"T-statistic: {t_stat:.3f}")
                print(f"P-value: {p_value:.3f}")
                print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

        print("\n✅ Analysis complete! Check the 'results/' directory for plots and data.")
        return results_summary

def main():
    parser = argparse.ArgumentParser(description="Analyze RL training results")
    parser.add_argument("--experiments", nargs="+", default=["curriculum", "hard_only"],
                       help="Experiments to analyze")
    parser.add_argument("--report", action="store_true",
                       help="Generate comprehensive report")
    args = parser.parse_args()

    analyzer = RLResultsAnalyzer()

    if args.report:
        analyzer.generate_comprehensive_report()
    else:
        # Quick analysis of specified experiments
        for exp in args.experiments:
            metrics = analyzer.calculate_performance_metrics(exp)
            if metrics:
                print(f"\n{exp.upper()} Performance:")
                print(f"  Episodes: {metrics['total_episodes']}")
                print(f"  Mean Reward: {metrics['mean_reward']:.3f}")
                print(f"  Std Reward: {metrics['std_reward']:.3f}")

if __name__ == "__main__":
    main()