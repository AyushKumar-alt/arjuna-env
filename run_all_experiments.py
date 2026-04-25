#!/usr/bin/env python3
"""
ARJUNA RL Project - Complete Execution Pipeline
Run all experiments and generate analysis for the RL course project.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"💥 {description} error: {e}")
        return False

def main():
    print("🚀 ARJUNA RL PROJECT - COMPLETE EXECUTION PIPELINE")
    print("=" * 60)

    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Phase 1: Enhanced PPO Experiments
    print("\n📚 PHASE 1: ENHANCED PPO TRAINING")
    print("-" * 40)

    experiments = [
        ("curriculum", "Auto-curriculum PPO"),
        ("hard_only", "Hard-only PPO (baseline)"),
    ]

    for exp_name, description in experiments:
        success = run_command([
            sys.executable, "train_rl_enhanced.py",
            "--timesteps", "3000",
            "--experiments", exp_name
        ], f"Training {description}")
        if not success:
            print(f"⚠️  Continuing despite failure in {exp_name}")

    # Phase 2: Baseline Algorithm Comparison
    print("\n🧪 PHASE 2: BASELINE ALGORITHM COMPARISON")
    print("-" * 40)

    success = run_command([
        sys.executable, "baseline_comparison.py",
        "--algorithms", "q_learning", "dqn",
        "--episodes", "2000",
        "--timesteps", "3000"
    ], "Running Q-Learning and DQN baselines")

    # Phase 3: Comprehensive Analysis
    print("\n📊 PHASE 3: COMPREHENSIVE ANALYSIS")
    print("-" * 40)

    success = run_command([
        sys.executable, "analyze_results.py", "--report"
    ], "Generating comprehensive analysis report")

    # Phase 4: Generate Enhanced Plots
    print("\n📈 PHASE 4: ENHANCED VISUALIZATION")
    print("-" * 40)

    # Run the existing plot script if it exists
    if Path("plot_rl_results.py").exists():
        success = run_command([
            sys.executable, "plot_rl_results.py"
        ], "Generating enhanced plots")

    # Phase 5: Export Dataset for Analysis
    print("\n💾 PHASE 5: DATASET EXPORT")
    print("-" * 40)

    if Path("export_dataset.py").exists():
        success = run_command([
            sys.executable, "export_dataset.py"
        ], "Exporting dataset for further analysis")

    # Final Summary
    print("\n🎉 EXECUTION COMPLETE!")
    print("=" * 60)
    print("📁 Check the following directories for results:")
    print("  • results/ - Training logs, summaries, and analysis")
    print("  • models/ - Saved model checkpoints")
    print("  • Generated plots and datasets")
    print("\n📋 Next steps:")
    print("  1. Review results in 'results/' directory")
    print("  2. Run 'python analyze_results.py --report' for detailed analysis")
    print("  3. Use results for your project report and presentation")

    # Quick results summary
    results_dir = Path("results")
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        print(f"\n📊 Found {len(csv_files)} result files:")
        for csv_file in csv_files:
            print(f"  • {csv_file.name}")

if __name__ == "__main__":
    main()