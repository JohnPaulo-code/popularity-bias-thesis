"""
Generate Thesis Figures from Analysis Data
==========================================
Creates:
1. Figure 1: NDCG vs APLT Trade-off (Pareto frontier)
2. Figure 2: Bar chart of APLT gain by algorithm and alpha
3. Figure 3: Efficiency comparison (APLT gain per 1% NDCG loss)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(
    r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml\enhanced_analysis\comparison\analysis_summary.csv")

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
})

COLORS = {
    'BinaryxQuAD': '#1f77b4',
    'SmoothxQuAD': '#ff7f0e',
    'FAIR': '#2ca02c',
}

# ============================================================================
# FIGURE 1: Trade-off Curves (NDCG Loss vs APLT Gain)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for algo in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
    algo_df = df[df['Algorithm'] == algo].sort_values('Alpha')

    # NDCG loss is negative, make positive for plotting
    ndcg_loss = -algo_df['NDCG_Change_%'].values
    aplt_gain = algo_df['APLT_Change_%'].values
    alphas = algo_df['Alpha'].values

    ax.plot(ndcg_loss, aplt_gain, 'o-', color=COLORS[algo],
            linewidth=2.5, markersize=8, label=algo)

    # Annotate alpha values
    for i, (x, y, a) in enumerate(zip(ndcg_loss, aplt_gain, alphas)):
        ax.annotate(f'α={a:.1f}', (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)

ax.set_xlabel('NDCG Loss (%)', fontsize=12)
ax.set_ylabel('APLT Gain (%)', fontsize=12)
ax.set_title('Accuracy-Fairness Trade-off: NDCG Loss vs APLT Gain', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(
    r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml\enhanced_analysis\plots\tradeoff_curves.png",
    dpi=300, bbox_inches='tight')
plt.close()
print("Saved: tradeoff_curves.png")

# ============================================================================
# FIGURE 2: APLT Gain by Algorithm and Alpha (Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

alphas = sorted(df['Alpha'].unique())
x = np.arange(len(alphas))
width = 0.25

for i, algo in enumerate(['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']):
    algo_df = df[df['Algorithm'] == algo].sort_values('Alpha')
    aplt_gains = algo_df['APLT_Change_%'].values

    bars = ax.bar(x + i * width, aplt_gains, width, label=algo, color=COLORS[algo])

    # Add value labels on bars
    for bar, val in zip(bars, aplt_gains):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Bias Level (α)', fontsize=12)
ax.set_ylabel('APLT Gain (%)', fontsize=12)
ax.set_title('Long-tail Coverage Improvement by Algorithm and Bias Level', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([f'α={a:.1f}' for a in alphas])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(
    r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml\enhanced_analysis\plots\aplt_gain_barchart.png",
    dpi=300, bbox_inches='tight')
plt.close()
print("Saved: aplt_gain_barchart.png")

# ============================================================================
# FIGURE 3: Efficiency (APLT gain per 1% NDCG loss)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for algo in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
    algo_df = df[df['Algorithm'] == algo].sort_values('Alpha')
    efficiencies = algo_df['Efficiency'].values
    alphas = algo_df['Alpha'].values

    ax.plot(alphas, efficiencies, 'o-', color=COLORS[algo],
            linewidth=2.5, markersize=8, label=algo)

ax.set_xlabel('Bias Level (α)', fontsize=12)
ax.set_ylabel('Efficiency (APLT gain / 1% NDCG loss)', fontsize=12)
ax.set_title('Mitigation Efficiency: How Much Fairness per Accuracy Cost', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='5x baseline')

plt.tight_layout()
plt.savefig(
    r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml\enhanced_analysis\plots\efficiency_curves.png",
    dpi=300, bbox_inches='tight')
plt.close()
print("Saved: efficiency_curves.png")

# ============================================================================
# PRINT FINAL TABLE FOR THESIS
# ============================================================================
print("\n" + "=" * 80)
print("THESIS-READY SUMMARY TABLE")
print("=" * 80)

# Create summary across alphas
summary_across = df.groupby('Algorithm').agg({
    'NDCG_Change_%': ['mean', 'std'],
    'APLT_Change_%': ['mean', 'std'],
    'ARP_Change_%': ['mean', 'std'],
    'ACLT_Change_%': ['mean', 'std'],
    'Efficiency': ['mean', 'std'],
}).round(1)

print(summary_across.to_string())

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR PRACTITIONERS")
print("=" * 80)
