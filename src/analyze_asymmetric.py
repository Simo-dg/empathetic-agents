"""
Analyze and visualize asymmetric empathy experiments.

Generates:
- Comparison plots showing how different empathy combinations perform
- Analysis of exploitation vs cooperation
- LaTeX tables for publication
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple


def load_asymmetric_results(log_dir: str = "logs") -> Dict[Tuple[float, float], Dict]:
    """Load results from asymmetric experiments."""
    summary_path = os.path.join(log_dir, "asymmetric_summary.csv")
    
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run train_asymmetric.py first.")
        return {}
    
    results = {}
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alpha1 = float(row['alpha_a1'])
            alpha2 = float(row['alpha_a2'])
            results[(alpha1, alpha2)] = {
                'coop_rate': float(row['last_100_avg_coop']),
                'total_welfare': float(row['last_100_avg_total_reward']),
                'reward_a1': float(row['last_100_avg_r1']),
                'reward_a2': float(row['last_100_avg_r2']),
                'time_to_coop': row['time_to_coop_threshold']
            }
    
    return results


def plot_asymmetric_analysis(results: Dict[Tuple[float, float], Dict], log_dir: str = "logs"):
    """Create comprehensive visualization of asymmetric results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data
    pairs = sorted(results.keys())
    labels = [f"({a1:.1f}, {a2:.1f})" for a1, a2 in pairs]
    
    coop_rates = [results[p]['coop_rate'] * 100 for p in pairs]
    total_welfare = [results[p]['total_welfare'] for p in pairs]
    reward_a1 = [results[p]['reward_a1'] for p in pairs]
    reward_a2 = [results[p]['reward_a2'] for p in pairs]
    
    # Identify symmetric vs asymmetric
    symmetric = [i for i, (a1, a2) in enumerate(pairs) if a1 == a2]
    asymmetric = [i for i, (a1, a2) in enumerate(pairs) if a1 != a2]
    
    x = np.arange(len(pairs))
    width = 0.6
    
    # Plot 1: Cooperation Rate
    colors = ['green' if i in symmetric else 'orange' for i in range(len(pairs))]
    bars1 = axes[0, 0].bar(x, coop_rates, width, color=colors, alpha=0.7)
    axes[0, 0].set_xlabel('Agent Pair (α₁, α₂)', fontsize=11)
    axes[0, 0].set_ylabel('Cooperation Rate (%)', fontsize=11)
    axes[0, 0].set_title('Cooperation Rate by Agent Pair', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    axes[0, 0].legend()
    
    # Plot 2: Social Welfare
    bars2 = axes[0, 1].bar(x, total_welfare, width, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=6, color='green', linestyle='--', label='Optimal (C,C)', alpha=0.7)
    axes[0, 1].axhline(y=2, color='red', linestyle='--', label='Nash (D,D)', alpha=0.7)
    axes[0, 1].set_xlabel('Agent Pair (α₁, α₂)', fontsize=11)
    axes[0, 1].set_ylabel('Total Welfare', fontsize=11)
    axes[0, 1].set_title('Social Welfare by Agent Pair', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend()
    
    # Plot 3: Individual Rewards Comparison
    x_rewards = np.arange(len(pairs))
    width_rewards = 0.35
    bars3a = axes[1, 0].bar(x_rewards - width_rewards/2, reward_a1, width_rewards, 
                            label='Agent 1', alpha=0.7, color='blue')
    bars3b = axes[1, 0].bar(x_rewards + width_rewards/2, reward_a2, width_rewards, 
                            label='Agent 2', alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Agent Pair (α₁, α₂)', fontsize=11)
    axes[1, 0].set_ylabel('Average Reward', fontsize=11)
    axes[1, 0].set_title('Individual Rewards by Agent', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x_rewards)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Exploitation Analysis (reward difference)
    reward_diff = [abs(reward_a1[i] - reward_a2[i]) for i in range(len(pairs))]
    bars4 = axes[1, 1].bar(x, reward_diff, width, color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Agent Pair (α₁, α₂)', fontsize=11)
    axes[1, 1].set_ylabel('Reward Difference |R₁ - R₂|', fontsize=11)
    axes[1, 1].set_title('Exploitation Level (Reward Asymmetry)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Symmetric pairs'),
        Patch(facecolor='orange', alpha=0.7, label='Asymmetric pairs')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    output_path = "plots/asymmetric_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Asymmetric analysis plot saved to: {os.path.abspath(output_path)}")
    
    plt.close()


def plot_cooperation_over_time(log_dir: str = "logs"):
    """Plot cooperation rate over time for different agent pairs (like cooperation.png)."""
    
    # Load all asymmetric experiment files
    files = [f for f in os.listdir(log_dir) if f.startswith('asymmetric_a1_') and f.endswith('.csv')]
    
    if not files:
        print("No asymmetric experiment files found.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for file in sorted(files):
        # Parse alphas from filename
        parts = file.replace('asymmetric_a1_', '').replace('.csv', '').split('_a2_')
        alpha1 = float(parts[0])
        alpha2 = float(parts[1])
        
        # Load episode data
        filepath = os.path.join(log_dir, file)
        df = pd.read_csv(filepath)
        
        # Compute moving average cooperation (window=100)
        window = 100
        coop_ma = df['coop_flag'].rolling(window=window, min_periods=1).mean() * 100
        
        # Determine line style and color
        if alpha1 == alpha2:
            # Symmetric pairs - solid lines
            linestyle = '-'
            if alpha1 == 1.0:
                color = 'red'
                label = f'Both Selfish (1.0, 1.0)'
            elif alpha1 <= 0.5:
                color = 'green'
                label = f'Both Empathic ({alpha1:.1f}, {alpha2:.1f})'
            else:
                color = 'blue'
                label = f'Symmetric ({alpha1:.1f}, {alpha2:.1f})'
        else:
            # Asymmetric pairs - dashed lines
            linestyle = '--'
            color = 'orange'
            label = f'Asymmetric ({alpha1:.1f}, {alpha2:.1f})'
        
        ax.plot(df['episode'], coop_ma, label=label, linestyle=linestyle, 
                color=color, alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cooperation Rate (%)', fontsize=12)
    ax.set_title('Cooperation Over Time: Symmetric vs Asymmetric Empathy', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    output_path = "plots/asymmetric_cooperation_over_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cooperation over time plot saved to: {os.path.abspath(output_path)}")
    plt.close()


def generate_asymmetric_latex_table(results: Dict[Tuple[float, float], Dict], 
                                    log_dir: str = "logs"):
    """Generate LaTeX table for asymmetric results."""
    
    output_path = "latex/asymmetric_results.tex"
    
    with open(output_path, 'w') as f:
        f.write("% Asymmetric Empathy Analysis - LaTeX Table\n\n")
        f.write("\\section{Asymmetric Empathy Experiments}\n\n")
        
        f.write("\\subsection{Motivation}\n")
        f.write("In real-world scenarios, agents may have different levels of empathy. ")
        f.write("We study how empathic agents interact with selfish agents to understand:\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item Can empathic agents be exploited by selfish agents?\n")
        f.write("\\item Do mixed pairs achieve cooperation?\n")
        f.write("\\item How does asymmetry affect social welfare?\n")
        f.write("\\end{itemize}\n\n")
        
        f.write("\\subsection{Results}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{cccccccc}\n")
        f.write("\\hline\n")
        f.write("Agent 1 & Agent 2 & Coop. & Social & Reward & Reward & Exploit. & Type \\\\\n")
        f.write("$\\alpha_1$ & $\\alpha_2$ & Rate & Welfare & Agent 1 & Agent 2 & Level & \\\\\n")
        f.write("\\hline\n")
        
        for (a1, a2), data in sorted(results.items()):
            pair_type = "Symmetric" if a1 == a2 else "Asymmetric"
            exploitation = abs(data['reward_a1'] - data['reward_a2'])
            
            f.write(f"{a1:.2f} & {a2:.2f} & "
                   f"{data['coop_rate']*100:.1f}\\% & "
                   f"{data['total_welfare']:.2f} & "
                   f"{data['reward_a1']:.2f} & "
                   f"{data['reward_a2']:.2f} & "
                   f"{exploitation:.2f} & "
                   f"{pair_type} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Results from asymmetric empathy experiments. Exploitation level "
               "measures reward difference between agents (higher = more exploitation).}\n")
        f.write("\\label{tab:asymmetric_results}\n")
        f.write("\\end{table}\n\n")
        
        # Analysis of key findings
        f.write("\\subsection{Key Findings}\n\n")
        f.write("\\begin{itemize}\n")
        
        # Find most exploited case
        asymmetric_pairs = {k: v for k, v in results.items() if k[0] != k[1]}
        if asymmetric_pairs:
            most_exploited = max(asymmetric_pairs.items(), 
                               key=lambda x: abs(x[1]['reward_a1'] - x[1]['reward_a2']))
            (a1_exp, a2_exp), data_exp = most_exploited
            
            f.write(f"\\item \\textbf{{Exploitation}}: Maximum exploitation occurs with "
                   f"$\\alpha_1={a1_exp:.2f}, \\alpha_2={a2_exp:.2f}$ "
                   f"(reward diff: {abs(data_exp['reward_a1'] - data_exp['reward_a2']):.2f})\n")
            
            # Check if empathic agent gets exploited
            if a1_exp > a2_exp and data_exp['reward_a1'] > data_exp['reward_a2']:
                f.write("\\item Selfish agents can exploit empathic agents in asymmetric settings\n")
            elif a1_exp < a2_exp and data_exp['reward_a2'] > data_exp['reward_a1']:
                f.write("\\item Selfish agents can exploit empathic agents in asymmetric settings\n")
        
        # Check cooperation in asymmetric pairs
        symmetric_coop = np.mean([v['coop_rate'] for k, v in results.items() if k[0] == k[1]])
        asymmetric_coop = np.mean([v['coop_rate'] for k, v in results.items() if k[0] != k[1]])
        
        f.write(f"\\item Average cooperation: Symmetric pairs {symmetric_coop*100:.1f}\\%, "
               f"Asymmetric pairs {asymmetric_coop*100:.1f}\\%\n")
        
        if asymmetric_coop < symmetric_coop * 0.8:
            f.write("\\item Empathy asymmetry significantly reduces cooperation\n")
        else:
            f.write("\\item Cooperation is relatively robust to empathy asymmetry\n")
        
        f.write("\\item This demonstrates the importance of mutual empathy for cooperation\n")
        f.write("\\end{itemize}\n")
    
    print(f"Asymmetric results table saved to: {os.path.abspath(output_path)}")
    return output_path


if __name__ == "__main__":
    from config import LOG_DIR
    
    print("\n### ASYMMETRIC EMPATHY ANALYSIS ###\n")
    
    results = load_asymmetric_results(LOG_DIR)
    
    if not results:
        print("No results found. Run train_asymmetric.py first.")
    else:
        print(f"Loaded {len(results)} agent pairs\n")
        
        print("Generating visualizations...")
        plot_asymmetric_analysis(results, LOG_DIR)
        
        print("Generating cooperation over time plot...")
        plot_cooperation_over_time(LOG_DIR)
        
        print("Generating LaTeX table...")
        generate_asymmetric_latex_table(results, LOG_DIR)
        
        print("\nAnalysis complete!")
