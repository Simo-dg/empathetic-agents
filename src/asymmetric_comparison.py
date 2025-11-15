"""
Compare empathic (0.2) and medium (0.8) agents in symmetric vs asymmetric settings.

Analyzes:
- How 0.2 agent performs with 0.2 partner vs 0.8 partner
- How 0.8 agent performs with 0.8 partner vs 0.2 partner
- Metrics: cooperation rate, total reward, regret, social welfare
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def load_episode_data(alpha1: float, alpha2: float, log_dir: str = "logs") -> pd.DataFrame:
    """Load episode-by-episode data for a specific agent pair."""
    filename = f"asymmetric_a1_{alpha1:.2f}_a2_{alpha2:.2f}.csv"
    filepath = os.path.join(log_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    return pd.read_csv(filepath)


def compute_metrics(df: pd.DataFrame, alpha1: float, alpha2: float) -> Dict:
    """Compute all metrics for the last 100 episodes."""
    if df is None or len(df) < 100:
        return None
    
    last_100 = df.tail(100)
    
    # Basic metrics
    coop_rate = last_100['coop_flag'].mean() * 100
    total_reward = last_100['total_reward'].mean()
    reward_a1 = last_100['r1'].mean()
    reward_a2 = last_100['r2'].mean()
    
    # Optimal rewards (mutual cooperation)
    optimal_r1 = 3.0
    optimal_r2 = 3.0
    optimal_total = 6.0
    
    # Individual regret (distance from optimal individual reward)
    individual_regret_a1 = optimal_r1 - reward_a1
    individual_regret_a2 = optimal_r2 - reward_a2
    
    # Social regret (distance from optimal social welfare)
    social_regret = optimal_total - total_reward
    
    # Fairness (reward difference)
    reward_diff = abs(reward_a1 - reward_a2)
    
    return {
        'alpha1': alpha1,
        'alpha2': alpha2,
        'coop_rate': coop_rate,
        'total_reward': total_reward,
        'reward_a1': reward_a1,
        'reward_a2': reward_a2,
        'individual_regret_a1': individual_regret_a1,
        'individual_regret_a2': individual_regret_a2,
        'social_regret': social_regret,
        'reward_diff': reward_diff,
        'avg_individual_regret': (individual_regret_a1 + individual_regret_a2) / 2
    }


def plot_comparison(log_dir: str = "logs"):
    """Create comparison plots for 0.2 and 0.8 agents."""
    
    # Load data for all relevant pairs (using available data)
    pairs = [
        (0.2, 0.2),   # Empathic with empathic
        (0.2, 0.8),   # Empathic with medium
        (0.8, 0.2),   # Medium with empathic
        (1.0, 1.0),   # Selfish with selfish (for reference)
    ]
    
    metrics_list = []
    for alpha1, alpha2 in pairs:
        df = load_episode_data(alpha1, alpha2, log_dir)
        metrics = compute_metrics(df, alpha1, alpha2)
        if metrics:
            metrics_list.append(metrics)
    
    if not metrics_list:
        print("No data available for comparison")
        return
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data for agent 0.2, 0.8, and 1.0
    agent_02_sym = next((m for m in metrics_list if m['alpha1'] == 0.2 and m['alpha2'] == 0.2), None)
    agent_02_asym = next((m for m in metrics_list if m['alpha1'] == 0.2 and m['alpha2'] == 0.8), None)
    agent_08_asym = next((m for m in metrics_list if m['alpha1'] == 0.8 and m['alpha2'] == 0.2), None)
    agent_10_sym = next((m for m in metrics_list if m['alpha1'] == 1.0 and m['alpha2'] == 1.0), None)
    
    # Plot 1: Cooperation Rate Comparison
    ax = axes[0, 0]
    x = [0, 1, 3, 4]
    heights = []
    labels = []
    colors = []
    
    if agent_02_sym:
        heights.append(agent_02_sym['coop_rate'])
        labels.append('0.2 with\n0.2')
        colors.append('#27ae60')
    if agent_02_asym:
        heights.append(agent_02_asym['coop_rate'])
        labels.append('0.2 with\n0.8')
        colors.append('#f39c12')
    if agent_08_asym:
        heights.append(agent_08_asym['coop_rate'])
        labels.append('0.8 with\n0.2')
        colors.append('#f39c12')
    if agent_10_sym:
        heights.append(agent_10_sym['coop_rate'])
        labels.append('1.0 with\n1.0')
        colors.append('#e74c3c')
    
    ax.bar(range(len(heights)), heights, color=colors, alpha=0.7, width=0.6)
    ax.set_ylabel('Cooperation Rate (%)', fontsize=11)
    ax.set_title('Cooperation Rate: Symmetric vs Asymmetric', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 2: Total Reward (Social Welfare)
    ax = axes[0, 1]
    heights = []
    if agent_02_sym:
        heights.append(agent_02_sym['total_reward'])
    if agent_02_asym:
        heights.append(agent_02_asym['total_reward'])
    if agent_08_asym:
        heights.append(agent_08_asym['total_reward'])
    if agent_10_sym:
        heights.append(agent_10_sym['total_reward'])
    
    ax.bar(range(len(heights)), heights, color=colors, alpha=0.7, width=0.6)
    ax.set_ylabel('Total Reward', fontsize=11)
    ax.set_title('Social Welfare: Symmetric vs Asymmetric', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=6, color='green', linestyle='--', alpha=0.5, label='Optimal (C,C)')
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Nash (D,D)')
    ax.legend(fontsize=9)
    
    # Plot 3: Social Regret
    ax = axes[1, 0]
    heights = []
    if agent_02_sym:
        heights.append(agent_02_sym['social_regret'])
    if agent_02_asym:
        heights.append(agent_02_asym['social_regret'])
    if agent_08_asym:
        heights.append(agent_08_asym['social_regret'])
    if agent_10_sym:
        heights.append(agent_10_sym['social_regret'])
    
    ax.bar(range(len(heights)), heights, color=colors, alpha=0.7, width=0.6)
    ax.set_ylabel('Social Regret', fontsize=11)
    ax.set_title('Social Regret: Distance from Optimal', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Individual Rewards (showing fairness)
    ax = axes[1, 1]
    x_pos = np.arange(len(labels))
    width = 0.35
    
    rewards_a1 = []
    rewards_a2 = []
    if agent_02_sym:
        rewards_a1.append(agent_02_sym['reward_a1'])
        rewards_a2.append(agent_02_sym['reward_a2'])
    if agent_02_asym:
        rewards_a1.append(agent_02_asym['reward_a1'])
        rewards_a2.append(agent_02_asym['reward_a2'])
    if agent_08_asym:
        rewards_a1.append(agent_08_asym['reward_a1'])
        rewards_a2.append(agent_08_asym['reward_a2'])
    if agent_10_sym:
        rewards_a1.append(agent_10_sym['reward_a1'])
        rewards_a2.append(agent_10_sym['reward_a2'])
    
    ax.bar(x_pos - width/2, rewards_a1, width, label='Agent 1', alpha=0.7, color='#3498db')
    ax.bar(x_pos + width/2, rewards_a2, width, label='Agent 2', alpha=0.7, color='#e74c3c')
    ax.set_ylabel('Individual Reward', fontsize=11)
    ax.set_title('Individual Rewards: Fairness Analysis', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = "plots/asymmetric_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {os.path.abspath(output_path)}")
    plt.close()


def plot_comparison_over_time(log_dir: str = "logs"):
    """Plot cooperation rate over time for symmetric vs asymmetric pairs."""
    
    # Load data for relevant pairs
    pairs = [
        (0.2, 0.2),   # Empathic with empathic
        (0.2, 0.8),   # Empathic with medium (asymmetric)
        (0.8, 0.2),   # Medium with empathic (asymmetric)
        (1.0, 1.0),   # Selfish with selfish
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        (0.2, 0.2): '#27ae60',  # Green
        (0.2, 0.8): '#f39c12',  # Orange
        (0.8, 0.2): '#9b59b6',  # Purple
        (1.0, 1.0): '#e74c3c',  # Red
    }
    
    labels_map = {
        (0.2, 0.2): 'Both Empathic (0.2, 0.2)',
        (0.2, 0.8): 'Empathic exploited (0.2, 0.8)',
        (0.8, 0.2): 'Medium exploits (0.8, 0.2)',
        (1.0, 1.0): 'Both Selfish (1.0, 1.0)',
    }
    
    linestyles = {
        (0.2, 0.2): '-',
        (0.2, 0.8): '--',
        (0.8, 0.2): '--',
        (1.0, 1.0): '-',
    }
    
    for alpha1, alpha2 in pairs:
        df = load_episode_data(alpha1, alpha2, log_dir)
        if df is None:
            continue
        
        # Compute moving average cooperation (window=100)
        window = 100
        coop_ma = df['coop_flag'].rolling(window=window, min_periods=window).mean() * 100
        
        ax.plot(df['episode'], coop_ma, 
                label=labels_map[(alpha1, alpha2)],
                color=colors[(alpha1, alpha2)],
                linestyle=linestyles[(alpha1, alpha2)],
                alpha=0.9, linewidth=2.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cooperation Rate (%)', fontsize=12)
    ax.set_title('Cooperation Evolution: Symmetric vs Asymmetric Empathy', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.set_ylim(-5, 105)
    ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    output_path = "plots/asymmetric_comparison_over_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cooperation over time plot saved to: {os.path.abspath(output_path)}")
    plt.close()


def generate_latex_table(log_dir: str = "logs"):
    """Generate comprehensive LaTeX table comparing symmetric vs asymmetric."""
    
    # Load data for all relevant pairs
    pairs = [
        (0.2, 0.2),
        (0.2, 0.8),
        (0.8, 0.2),
        (1.0, 1.0),
    ]
    
    metrics_list = []
    for alpha1, alpha2 in pairs:
        df = load_episode_data(alpha1, alpha2, log_dir)
        metrics = compute_metrics(df, alpha1, alpha2)
        if metrics:
            metrics_list.append(metrics)
    
    if not metrics_list:
        print("No data available for table generation")
        return
    
    output_path = "latex/asymmetric_comparison.tex"
    
    with open(output_path, 'w') as f:
        f.write("% Asymmetric vs Symmetric Empathy Comparison\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of Agent Performance: Symmetric vs Asymmetric Pairings}\n")
        f.write("\\label{tab:asymmetric_comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write("Agent Pair & Coop. & Total & Reward & Reward & Social & Reward \\\\\n")
        f.write("$(\\alpha_1, \\alpha_2)$ & Rate (\\%) & Reward & A1 & A2 & Regret & Diff. \\\\\n")
        f.write("\\hline\n")
        
        for m in metrics_list:
            # Determine if symmetric
            is_sym = m['alpha1'] == m['alpha2']
            prefix = "\\textbf{" if is_sym else ""
            suffix = "}" if is_sym else ""
            
            f.write(f"{prefix}({m['alpha1']:.1f}, {m['alpha2']:.1f}){suffix} & ")
            f.write(f"{prefix}{m['coop_rate']:.1f}{suffix} & ")
            f.write(f"{prefix}{m['total_reward']:.2f}{suffix} & ")
            f.write(f"{prefix}{m['reward_a1']:.2f}{suffix} & ")
            f.write(f"{prefix}{m['reward_a2']:.2f}{suffix} & ")
            f.write(f"{prefix}{m['social_regret']:.2f}{suffix} & ")
            f.write(f"{prefix}{m['reward_diff']:.2f}{suffix} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Add detailed regret table
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Regret Analysis: Individual vs Social}\n")
        f.write("\\label{tab:regret_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Agent Pair & Regret & Regret & Avg. Individual & Social \\\\\n")
        f.write("$(\\alpha_1, \\alpha_2)$ & A1 & A2 & Regret & Regret \\\\\n")
        f.write("\\hline\n")
        
        for m in metrics_list:
            is_sym = m['alpha1'] == m['alpha2']
            prefix = "\\textbf{" if is_sym else ""
            suffix = "}" if is_sym else ""
            
            f.write(f"{prefix}({m['alpha1']:.1f}, {m['alpha2']:.1f}){suffix} & ")
            f.write(f"{prefix}{m['individual_regret_a1']:.3f}{suffix} & ")
            f.write(f"{prefix}{m['individual_regret_a2']:.3f}{suffix} & ")
            f.write(f"{prefix}{m['avg_individual_regret']:.3f}{suffix} & ")
            f.write(f"{prefix}{m['social_regret']:.3f}{suffix} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX comparison tables saved to: {os.path.abspath(output_path)}")


def main():
    """Run the asymmetric comparison analysis."""
    print("\n### ASYMMETRIC COMPARISON ANALYSIS ###\n")
    print("Comparing agent 0.2 and 0.8 in symmetric vs asymmetric settings\n")
    
    # Create plots
    print("Generating comparison plots...")
    plot_comparison()
    
    # Create over-time plot
    print("Generating cooperation over time plot...")
    plot_comparison_over_time()
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_table()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
