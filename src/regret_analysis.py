"""
Regret Analysis Module

Computes and visualizes regret metrics:
- Social regret: difference between optimal social welfare and actual total reward
- Individual regret: difference between best response and actual reward for each agent
- Cumulative regret: sum of all regrets over time
- Average regret: cumulative regret / number of episodes

Individual regret measures if agents play best response to opponents' actions.
Social regret measures distance from Pareto optimal outcome.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from config import PD_PAYOFFS, ACTIONS

def per_agent_instant_regret(a1, a2, payoffs):
    # realized rewards
    r1, r2 = payoffs[(a1, a2)]
    # best replies to realized opponent action
    best_r1 = max(payoffs[(ai, a2)][0] for ai in ACTIONS)
    best_r2 = max(payoffs[(a1, aj)][1] for aj in ACTIONS)
    return (best_r1 - r1, best_r2 - r2)


def compute_regret(log_file: str, optimal_welfare: float = 6.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute regret metrics from episode log.
    
    Args:
        log_file: Path to per-episode CSV log
        optimal_welfare: Maximum possible social welfare (default: 6 for C,C in PD)
    
    Returns:
        Tuple of (instantaneous_regret, cumulative_regret, average_regret)
    """
    episodes = []
    total_rewards = []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            total_rewards.append(float(row['total_reward']))
    
    # Instantaneous regret: optimal - actual at each episode
    instantaneous_regret = [optimal_welfare - r for r in total_rewards]
    
    # Cumulative regret: sum of all regrets up to episode t
    cumulative_regret = np.cumsum(instantaneous_regret).tolist()
    
    # Average regret: cumulative / episode number
    average_regret = [cumulative_regret[i] / (i + 1) for i in range(len(cumulative_regret))]
    
    return instantaneous_regret, cumulative_regret, average_regret


def save_regret_data(log_file: str, output_file: str, optimal_welfare: float = 6.0):
    """
    Compute regret and save to CSV.
    
    Args:
        log_file: Path to input per-episode log
        output_file: Path to output regret CSV
        optimal_welfare: Maximum possible social welfare
    """
    inst_regret, cum_regret, avg_regret = compute_regret(log_file, optimal_welfare)
    
    # Read original data
    episodes = []
    empathy_alphas = []
    total_rewards = []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            empathy_alphas.append(float(row['alpha_emp']))
            total_rewards.append(float(row['total_reward']))
    
    # Write regret data
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'alpha_emp', 'total_reward', 'optimal_welfare',
            'instantaneous_regret', 'cumulative_regret', 'average_regret'
        ])
        
        for i in range(len(episodes)):
            writer.writerow([
                episodes[i], empathy_alphas[i], total_rewards[i], optimal_welfare,
                inst_regret[i], cum_regret[i], avg_regret[i]
            ])
    
    return output_file


def plot_regret_analysis(regret_files: Dict[float, str], log_dir: str = "logs"):
    """
    Create comprehensive regret visualization with both social and individual regret.
    
    Args:
        regret_files: Dict mapping empathy_alpha -> per-episode CSV file path
        log_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(regret_files)))
    
    for idx, (alpha, filepath) in enumerate(sorted(regret_files.items(), reverse=True)):
        episodes = []
        social_regret = []
        cum_social = []
        avg_social = []
        regret_a1 = []
        regret_a2 = []
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes.append(int(row['episode']))
                social_regret.append(float(row['social_regret']))
                cum_social.append(float(row['cumulative_social_regret']))
                avg_social.append(float(row['avg_social_regret']))
                regret_a1.append(float(row['regret_a1']))
                regret_a2.append(float(row['regret_a2']))
        
        label = f'α={alpha:.2f}'
        color = colors[idx]
        
        # Plot 1: Instantaneous Social Regret (smoothed)
        window = 100
        if len(social_regret) >= window:
            smoothed = np.convolve(social_regret, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(episodes[window-1:], smoothed, label=label, color=color, linewidth=2)
        else:
            axes[0, 0].plot(episodes, social_regret, label=label, color=color, linewidth=2, alpha=0.5)
        
        # Plot 2: Cumulative Social Regret
        axes[0, 1].plot(episodes, cum_social, label=label, color=color, linewidth=2)
        
        # Plot 3: Average Social Regret
        axes[0, 2].plot(episodes, avg_social, label=label, color=color, linewidth=2)
        
        # Plot 4: Individual Regret Agent 1 (smoothed)
        if len(regret_a1) >= window:
            smoothed_a1 = np.convolve(regret_a1, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(episodes[window-1:], smoothed_a1, label=label, color=color, linewidth=2)
        
        # Plot 5: Individual Regret Agent 2 (smoothed)
        if len(regret_a2) >= window:
            smoothed_a2 = np.convolve(regret_a2, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(episodes[window-1:], smoothed_a2, label=label, color=color, linewidth=2)
        
        # Plot 6: Average Individual Regret (both agents)
        avg_individual = [(regret_a1[i] + regret_a2[i])/2 for i in range(len(regret_a1))]
        if len(avg_individual) >= window:
            smoothed_avg = np.convolve(avg_individual, np.ones(window)/window, mode='valid')
            axes[1, 2].plot(episodes[window-1:], smoothed_avg, label=label, color=color, linewidth=2)
    
    # Configure Plot 1: Instantaneous Social Regret
    axes[0, 0].set_xlabel('Episode', fontsize=11)
    axes[0, 0].set_ylabel('Social Regret', fontsize=11)
    axes[0, 0].set_title('Social Regret (100-ep moving avg)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Configure Plot 2: Cumulative Social Regret
    axes[0, 1].set_xlabel('Episode', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative Social Regret', fontsize=11)
    axes[0, 1].set_title('Cumulative Social Regret', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Configure Plot 3: Average Social Regret
    axes[0, 2].set_xlabel('Episode', fontsize=11)
    axes[0, 2].set_ylabel('Avg Social Regret', fontsize=11)
    axes[0, 2].set_title('Average Social Regret', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Configure Plot 4: Individual Regret Agent 1
    axes[1, 0].set_xlabel('Episode', fontsize=11)
    axes[1, 0].set_ylabel('Regret Agent 1', fontsize=11)
    axes[1, 0].set_title('Individual Regret - Agent 1 (vs Best Response)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Configure Plot 5: Individual Regret Agent 2
    axes[1, 1].set_xlabel('Episode', fontsize=11)
    axes[1, 1].set_ylabel('Regret Agent 2', fontsize=11)
    axes[1, 1].set_title('Individual Regret - Agent 2 (vs Best Response)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Configure Plot 6: Average Individual Regret
    axes[1, 2].set_xlabel('Episode', fontsize=11)
    axes[1, 2].set_ylabel('Avg Individual Regret', fontsize=11)
    axes[1, 2].set_title('Average Individual Regret (Both Agents)', fontsize=12, fontweight='bold')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_path = "plots/regret_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Regret analysis plot saved to: {os.path.abspath(output_path)}")
    
    plt.close()


def generate_regret_summary_table(log_dir: str = "logs"):
    """
    Generate LaTeX table summarizing both social and individual regret metrics.
    
    Args:
        log_dir: Directory containing logs and to save LaTeX file
    """
    from config import ALPHAS
    
    output_path = "latex/regret_summary.tex"
    
    summary_data = []
    
    for alpha in ALPHAS:
        filepath = os.path.join(log_dir, f"per_episode_alpha_{alpha:.2f}.csv")
        
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            continue
        
        # Get final metrics
        final_row = rows[-1]
        final_cum_social = float(final_row['cumulative_social_regret'])
        final_avg_social = float(final_row['avg_social_regret'])
        final_cum_a1 = float(final_row['cumulative_regret_a1'])
        final_cum_a2 = float(final_row['cumulative_regret_a2'])
        
        # Get last 100 episodes average
        last_100 = rows[-100:] if len(rows) >= 100 else rows
        last_100_social = np.mean([float(r['avg_social_regret']) for r in last_100])
        last_100_a1 = np.mean([float(r['avg_regret_a1']) for r in last_100])
        last_100_a2 = np.mean([float(r['avg_regret_a2']) for r in last_100])
        last_100_individual = (last_100_a1 + last_100_a2) / 2
        
        summary_data.append({
            'alpha': alpha,
            'final_cum_social': final_cum_social,
            'final_avg_social': final_avg_social,
            'last_100_social': last_100_social,
            'last_100_a1': last_100_a1,
            'last_100_a2': last_100_a2,
            'last_100_individual': last_100_individual
        })
    
    with open(output_path, 'w') as f:
        f.write("% Regret Analysis Summary - LaTeX Table\n\n")
        f.write("\\section{Regret Analysis}\n\n")
        
        f.write("\\subsection{Definition}\n")
        f.write("Two types of regret are measured:\n")
        f.write("\\begin{itemize}\n")
        f.write("\\item \\textbf{Social Regret}: $r^{social}_t = R^* - (R^1_t + R^2_t)$ where $R^* = 6$ (optimal C,C outcome)\n")
        f.write("  \\begin{itemize}\n")
        f.write("  \\item Measures distance from Pareto optimal outcome\n")
        f.write("  \\item Evaluates collective welfare\n")
        f.write("  \\end{itemize}\n")
        f.write("\\item \\textbf{Individual Regret}: $r^i_t = BR^i(a^{-i}_t) - R^i_t$ where $BR^i$ is best response to opponent's action\n")
        f.write("  \\begin{itemize}\n")
        f.write("  \\item Measures if agent plays optimally given opponent's action\n")
        f.write("  \\item Non-zero means suboptimal individual play\n")
        f.write("  \\end{itemize}\n")
        f.write("\\end{itemize}\n\n")
        
        # Social Regret Table
        f.write("\\subsection{Social Regret Summary}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\hline\n")
        f.write("Empathy & Final Cumulative & Final Average & Last 100 Avg \\\\\n")
        f.write("$\\alpha$ & Social Regret & Social Regret & Social Regret \\\\\n")
        f.write("\\hline\n")
        
        for data in summary_data:
            f.write(f"{data['alpha']:.2f} & "
                   f"{data['final_cum_social']:.1f} & "
                   f"{data['final_avg_social']:.3f} & "
                   f"{data['last_100_social']:.3f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Social regret measures distance from optimal social welfare (6 for C,C).}\n")
        f.write("\\label{tab:social_regret}\n")
        f.write("\\end{table}\n\n")
        
        # Individual Regret Table
        f.write("\\subsection{Individual Regret Summary}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{ccccc}\n")
        f.write("\\hline\n")
        f.write("Empathy & Agent 1 & Agent 2 & Average & Interpretation \\\\\n")
        f.write("$\\alpha$ & Regret & Regret & Individual Regret & \\\\\n")
        f.write("\\hline\n")
        
        for data in summary_data:
            interpretation = "Cooperative" if data['last_100_individual'] > 0 else "Best Response"
            f.write(f"{data['alpha']:.2f} & "
                   f"{data['last_100_a1']:.3f} & "
                   f"{data['last_100_a2']:.3f} & "
                   f"{data['last_100_individual']:.3f} & "
                   f"{interpretation} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Individual regret (last 100 episodes avg). Positive regret means agents sacrifice "
               "personal gain for cooperation. Zero regret means playing best response.}\n")
        f.write("\\label{tab:individual_regret}\n")
        f.write("\\end{table}\n\n")
        
        f.write("\\subsection{Key Insights}\n\n")
        f.write("\\begin{itemize}\n")
        
        # Find best social welfare
        best_social = min(summary_data, key=lambda x: x['last_100_social'])
        worst_social = max(summary_data, key=lambda x: x['last_100_social'])
        
        f.write(f"\\item \\textbf{{Social Regret}}: Best $\\alpha={best_social['alpha']:.2f}$ "
               f"(regret={best_social['last_100_social']:.3f}), "
               f"Worst $\\alpha={worst_social['alpha']:.2f}$ "
               f"(regret={worst_social['last_100_social']:.3f})\n")
        
        f.write(f"\\item \\textbf{{Individual Regret}}: Positive individual regret indicates agents "
               f"choose cooperation over selfish best response\n")
        
        f.write("\\item Empathic agents accept individual regret to achieve lower social regret\n")
        f.write("\\item This demonstrates the trade-off between individual optimality and collective welfare\n")
        f.write("\\end{itemize}\n")
    
    print(f"Regret summary table saved to: {os.path.abspath(output_path)}")
    return output_path


def analyze_all_regret(log_dir: str = "logs"):
    """
    Analyze regret for all empathy levels.
    
    Args:
        log_dir: Directory containing per-episode logs
    """
    from config import ALPHAS
    
    print("\n" + "="*70)
    print("REGRET ANALYSIS - Social and Individual")
    print("="*70 + "\n")
    
    regret_files = {}
    
    # Process each empathy level
    for alpha in ALPHAS:
        log_file = os.path.join(log_dir, f"per_episode_alpha_{alpha:.2f}.csv")
        
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found, skipping α={alpha}")
            continue
        
        print(f"Processing regret for α={alpha:.2f}...")
        regret_files[alpha] = log_file
    
    if not regret_files:
        print("\nNo log files found. Run train.py first.")
        return
    
    print("\nGenerating visualizations...")
    plot_regret_analysis(regret_files, log_dir)
    
    print("\nGenerating summary table...")
    generate_regret_summary_table(log_dir)
    
    print("\n" + "="*70)
    print("Regret analysis complete!")
    print("="*70)


if __name__ == "__main__":
    from config import LOG_DIR
    
    analyze_all_regret(LOG_DIR)
