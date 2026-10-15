"""
Sensitivity Plots — BookCrossing
==================================
Adapted from sensitivity_plots_ml.py for BookCrossing dataset.
Generates all visualizations for the BookCrossing sensitivity study.

Changes from ML version:
    - Updated titles to reference BookCrossing
    - Dataset-specific constants preserved
    - Nash parameters loaded from BC directory
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# =============================================================================
# GLOBAL STYLE
# =============================================================================

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'legend.fontsize':   10,
    'figure.titlesize':  14,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS = {
    'BPR_baseline': '#333333',
    'BinaryxQuAD':  '#1f77b4',
    'SmoothxQuAD':  '#ff7f0e',
    'FAIR':         '#2ca02c',
}

LABELS = {
    'BinaryxQuAD':  'Binary xQuAD (λ)',
    'SmoothxQuAD':  'Smooth xQuAD (λ)',
    'FAIR':         'FA*IR (τ)',
    'BPR_baseline': 'BPR Baseline',
}

ALPHA_LABELS = ['α=1.0', 'α=1.2', 'α=1.5', 'α=2.0', 'α=2.5']
ALPHAS_FLOAT = [1.0, 1.2, 1.5, 2.0, 2.5]
ALPHA_STRS   = ['1.0', '1.2', '1.5', '2.0', '2.5']

# Default placeholders — replaced at runtime if nash_optimal_params.json is found.
FIXED_LAMBDAS = {'BinaryxQuAD': 0.5, 'SmoothxQuAD': 0.6, 'FAIR': 0.7}

K = 10


def load_fixed_lambdas(nash_json_path: str = None) -> dict:
    """
    Load consensus λ* per method from nash_optimal_params.json produced by
    nash_parameter_selection.py.  Takes the mode across alpha levels as the
    single reference value used for vertical lines in figures.
    Falls back to FIXED_LAMBDAS defaults if the file is absent.
    """
    if nash_json_path is None or not os.path.exists(nash_json_path):
        return dict(FIXED_LAMBDAS)

    with open(nash_json_path) as f:
        nash = json.load(f)

    from collections import Counter
    label_to_key = {
        'Binary xQuAD': 'BinaryxQuAD',
        'Smooth xQuAD': 'SmoothxQuAD',
        'FA*IR':        'FAIR',
    }
    result = dict(FIXED_LAMBDAS)
    for label, key in label_to_key.items():
        if label in nash:
            vals = list(nash[label].values())
            # Filter out None values (ineffective methods)
            vals = [v for v in vals if v is not None]
            if vals:
                result[key] = Counter(vals).most_common(1)[0][0]
    return result


# =============================================================================
# FIGURE 3 — BPR Baseline Trend
# =============================================================================

def plot_figure3_baseline(agg: dict, output_dir: str) -> None:
    """
    Dual-axis line plot: NDCG@10 (left axis) and APLT@10 (right axis)
    as α increases. Error bands = ±1 std across five seeds.
    """
    ndcg_means, ndcg_stds = [], []
    aplt_means, aplt_stds = [], []

    for a in ALPHA_STRS:
        b = agg['BPR_baseline'][a]
        ndcg_means.append(b['NDCG@10']['mean'])
        ndcg_stds.append( b['NDCG@10']['std'])
        aplt_means.append(b['APLT@10']['mean'])
        aplt_stds.append( b['APLT@10']['std'])

    ndcg_means = np.array(ndcg_means)
    ndcg_stds  = np.array(ndcg_stds)
    aplt_means = np.array(aplt_means)
    aplt_stds  = np.array(aplt_stds)
    x          = np.array(ALPHAS_FLOAT)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(x, ndcg_means, 'o-', color='#1f77b4', linewidth=2.5,
             markersize=8, label='NDCG@10 (accuracy)')
    ax1.fill_between(x, ndcg_means - ndcg_stds, ndcg_means + ndcg_stds,
                     alpha=0.15, color='#1f77b4')

    ax2.plot(x, aplt_means, 's--', color='#d62728', linewidth=2.5,
             markersize=8, label='APLT@10 (fairness)')
    ax2.fill_between(x, aplt_means - aplt_stds, aplt_means + aplt_stds,
                     alpha=0.15, color='#d62728')

    ax1.set_xlabel('Bias Level (α)', fontsize=12)
    ax1.set_ylabel('NDCG@10', fontsize=12, color='#1f77b4')
    ax2.set_ylabel('APLT@10', fontsize=12, color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ALPHA_LABELS)

    for xi, yn, ya in zip(x, ndcg_means, aplt_means):
        ax1.annotate(f'{yn:.3f}', (xi, yn),
                     textcoords='offset points', xytext=(0, 10),
                     ha='center', fontsize=9, color='#1f77b4')
        ax2.annotate(f'{ya:.3f}', (xi, ya),
                     textcoords='offset points', xytext=(0, -16),
                     ha='center', fontsize=9, color='#d62728')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2,
               loc='center right', frameon=True, framealpha=0.9)

    ax1.set_title(
        'Figure 3: BookCrossing — BPR Baseline Accuracy vs Fairness Across Bias Levels\n'
        '(mean ± 1 std across five random seeds)',
        fontsize=13, fontweight='bold', pad=12
    )
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(output_dir, 'figure3_baseline_trend.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 4 — Lambda Sensitivity Curves
# =============================================================================

def plot_figure4_lambda_curves(agg: dict, output_dir: str) -> None:
    """
    2 rows × 2 cols grid:
      Row 0: Binary xQuAD  — NDCG@10 vs λ  |  APLT@10 vs λ
      Row 1: Smooth xQuAD  — NDCG@10 vs λ  |  APLT@10 vs λ
    Each subplot shows all five α levels as colour-coded lines.
    Dotted horizontal lines = BPR baseline per α.
    Red dashed vertical line = fixed λ*.
    """
    lambda_grid  = np.array(agg['lambda_grid'])
    algos        = ['BinaryxQuAD', 'SmoothxQuAD']
    metrics      = ['NDCG@10', 'APLT@10']
    metric_labels= ['NDCG@10 (Accuracy)', 'APLT@10 (Long-tail Coverage)']
    alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ALPHA_STRS)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')
    fig.suptitle(
        'Figure 4: BookCrossing — Lambda Sensitivity Curves\n'
        '(mean ± 1 std across five seeds; vertical line = fixed λ*)',
        fontsize=13, fontweight='bold', y=1.01
    )

    for row, algo in enumerate(algos):
        lam_star = FIXED_LAMBDAS[algo]

        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row][col]

            for i, (a_str, a_float) in enumerate(zip(ALPHA_STRS, ALPHAS_FLOAT)):
                mean_arr = np.array(agg[algo][a_str][metric]['mean'])
                std_arr  = np.array(agg[algo][a_str][metric]['std'])
                b_val    = agg['BPR_baseline'][a_str][metric]['mean']

                ax.plot(lambda_grid, mean_arr, 'o-', color=alpha_colors[i],
                        linewidth=1.8, markersize=5, label=f'α={a_float}', alpha=0.9)
                ax.fill_between(lambda_grid, mean_arr - std_arr, mean_arr + std_arr,
                                alpha=0.08, color=alpha_colors[i])
                ax.axhline(b_val, color=alpha_colors[i],
                           linestyle=':', linewidth=1, alpha=0.5)

            ax.axvline(lam_star, color='red', linestyle='--',
                       linewidth=1.8, alpha=0.8, label=f'λ* = {lam_star}')

            ax.set_xlabel('λ (fairness weight)', fontsize=11)
            ax.set_ylabel(f'{mlabel}\n↑ higher is better', fontsize=10)
            ax.set_title(f'{algo}', fontsize=12, fontweight='bold')
            ax.set_xticks(lambda_grid)
            ax.set_xticklabels([f'{v:.1f}' for v in lambda_grid],
                               rotation=45, fontsize=8)
            ax.grid(True, alpha=0.25, linestyle='--')

            if row == 0 and col == 1:
                ax.legend(loc='upper right', fontsize=8, frameon=True, ncol=2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figure4_lambda_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 5 — FA*IR Tau Sensitivity Curves
# =============================================================================

def plot_figure5_tau_curves(agg: dict, output_dir: str) -> None:
    """
    1 row × 2 cols:
      Left:  NDCG@10 vs τ for all α
      Right: APLT@10 vs τ for all α
    Grey shaded region marks the inactive zone τ = 0–0.2.
    Vertical lines mark the activation point (τ=0.3) and τ*=0.5.
    """
    tau_grid     = np.array(agg['tau_grid'])
    alpha_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(ALPHA_STRS)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        'Figure 5: BookCrossing — FA*IR Tau Sensitivity Curves\n'
        '(mean ± 1 std across five seeds; shaded region = inactive threshold zone)',
        fontsize=13, fontweight='bold'
    )

    for col, (metric, mlabel) in enumerate([
        ('NDCG@10', 'NDCG@10 (Accuracy) — ↑ higher is better'),
        ('APLT@10', 'APLT@10 (Long-tail Coverage) — ↑ higher is better'),
    ]):
        ax = axes[col]

        for i, (a_str, a_float) in enumerate(zip(ALPHA_STRS, ALPHAS_FLOAT)):
            mean_arr = np.array(agg['FAIR'][a_str][metric]['mean'])
            std_arr  = np.array(agg['FAIR'][a_str][metric]['std'])

            ax.plot(tau_grid, mean_arr, 's-', color=alpha_colors[i],
                    linewidth=1.8, markersize=6,
                    label=f'α={a_float}', alpha=0.9)
            ax.fill_between(tau_grid, mean_arr - std_arr, mean_arr + std_arr,
                            alpha=0.08, color=alpha_colors[i])

        ax.axvspan(0.0, 0.2, alpha=0.08, color='grey',
                   label='Inactive zone (τ=0–0.2)')
        ax.axvline(0.3, color='grey', linestyle=':', linewidth=1.5, alpha=0.7,
                   label='Activation (τ=0.3)')
        ax.axvline(0.5, color='red',  linestyle='--', linewidth=1.8, alpha=0.85,
                   label='τ* = 0.5')

        ax.set_xlabel('τ (minimum protected proportion)', fontsize=11)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f'FA*IR — {metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(tau_grid)
        ax.set_xticklabels([f'{v:.1f}' for v in tau_grid])
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc='best', fontsize=8, frameon=True, ncol=2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figure5_tau_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 6 — Key Finding: Optimal Lambda Invariance
# =============================================================================

def plot_figure6_key_finding(agg: dict, output_dir: str) -> None:
    """
    Composite-optimal λ* per method per α with ±1 std error bars.
    Near-flat lines across α levels are the core invariance finding.
    Horizontal dashed lines mark the consensus λ* for each method.
    """
    x   = np.array(ALPHAS_FLOAT)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ['BinaryxQuAD', 'SmoothxQuAD']:
        opt_lambdas, opt_stds = [], []
        lam_grid = np.array(agg['lambda_grid'])

        for a_str in ALPHA_STRS:
            ndcg     = np.array(agg[algo][a_str]['NDCG@10']['mean'])
            aplt     = np.array(agg[algo][a_str]['APLT@10']['mean'])
            ndcg_std = np.array(agg[algo][a_str]['NDCG@10']['std'])

            ndcg_n = (ndcg - ndcg.min()) / (ndcg.max() - ndcg.min() + 1e-10)
            aplt_n = (aplt - aplt.min()) / (aplt.max() - aplt.min() + 1e-10)
            comp   = (ndcg_n + aplt_n) / 2

            best_idx = int(np.argmax(comp))
            opt_lambdas.append(lam_grid[best_idx])
            opt_stds.append(float(ndcg_std[best_idx]))

        ax.errorbar(x, opt_lambdas, yerr=opt_stds,
                    fmt='o-', color=COLORS[algo],
                    linewidth=2.5, markersize=9, capsize=6, capthick=2,
                    markeredgecolor='white', markeredgewidth=1.5,
                    label=LABELS[algo])

        consensus = FIXED_LAMBDAS[algo]
        ax.axhline(consensus, color=COLORS[algo],
                   linestyle='--', linewidth=1.2, alpha=0.5)
        ax.annotate(
            f'λ* = {consensus}',
            xy=(ALPHAS_FLOAT[-1], consensus),
            xytext=(8, 4), textcoords='offset points',
            fontsize=10, color=COLORS[algo], fontweight='bold'
        )

    ax.axhline(0.5, color=COLORS['FAIR'], linestyle='-.', linewidth=2, alpha=0.7,
               label='FA*IR τ* = 0.5 (fixed)')

    ax.set_xlabel('Bias Level (α)', fontsize=12)
    ax.set_ylabel('Optimal Parameter (λ* or τ*)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ALPHA_LABELS, fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_title(
        'Figure 6: BookCrossing — Optimal Parameter is Invariant to Bias Level\n'
        'λ* remains stable across α = 1.0 → 2.5 for all three methods',
        fontsize=13, fontweight='bold', pad=14
    )

    props = dict(boxstyle='round,pad=0.4', facecolor='#fffbe6',
                 edgecolor='#ccaa00', alpha=0.9)
    ax.text(0.03, 0.12,
            'Finding: Practitioners do not need to\n'
            're-tune λ as dataset bias intensifies.\n'
            'A single calibration suffices.',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    path = os.path.join(output_dir, 'figure6_key_finding_lambda.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_figure8_significance_heatmap(stats_df: pd.DataFrame,
                                      output_dir: str) -> None:
    """
    Heatmap: rows = methods, cols = α levels.
    Cell value = proportion of seeds with p < 0.05 (paired t-test).
    Reads statistical_tests.csv (best-λ per seed).
    """
    alphas = sorted(stats_df['alpha'].unique())
    algos  = ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']
    matrix = np.zeros((len(algos), len(alphas)))

    for i, algo in enumerate(algos):
        for j, alpha in enumerate(alphas):
            sub = stats_df[(stats_df['algorithm'] == algo) &
                           (stats_df['alpha'] == alpha)]
            if not sub.empty:
                matrix[i, j] = sub['sig_ttest_0.05'].mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        matrix,
        annot=True, fmt='.0%',
        xticklabels=[f'α={a}' for a in alphas],
        yticklabels=['Binary xQuAD', 'Smooth xQuAD', 'FA*IR'],
        cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
        cbar_kws={'label': 'Proportion of seeds significant (p < 0.05)',
                  'shrink': 0.8},
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 13, 'fontweight': 'bold'}
    )
    ax.set_title(
        "Figure 8: BookCrossing — Statistical Significance\n"
        "Paired t-test on Per-user NDCG@10 (green = consistently significant)",
        fontsize=12, fontweight='bold', pad=12
    )
    ax.set_xlabel('Bias Level (α)', fontsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11, rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'figure8_significance_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# SensitivityPlotter CLASS  (imported by stage2_run_mitigation.py)
# =============================================================================

class SensitivityPlotter:
    """
    Thin wrapper around the standalone plot functions so that
    stage2_run_mitigation_bc.py can import and call a class interface.
    """

    def __init__(self, output_dir: str, k: int = 10,
                 nash_json_path: str = None):
        self.output_dir = output_dir
        self.k          = k
        os.makedirs(output_dir, exist_ok=True)

        # Update the module-level FIXED_LAMBDAS if Nash params are available
        global FIXED_LAMBDAS
        FIXED_LAMBDAS = load_fixed_lambdas(nash_json_path)

    def plot_tradeoff_curves(self, aggregated: dict) -> None:
        """Figures 4 (xQuAD lambda curves) and 5 (FA*IR tau curves)."""
        plot_figure4_lambda_curves(aggregated, self.output_dir)
        plot_figure5_tau_curves(aggregated, self.output_dir)

    def plot_dashboard(self, aggregated: dict) -> None:
        """Figure 3 — BPR baseline trend across bias levels."""
        plot_figure3_baseline(aggregated, self.output_dir)

    def plot_key_finding(self, aggregated: dict) -> None:
        """Figure 6 — optimal parameter invariance finding."""
        plot_figure6_key_finding(aggregated, self.output_dir)

    def plot_all(self, aggregated: dict,
                 fixed_df=None, stats_df=None) -> None:
        """Generate all figures."""
        self.plot_dashboard(aggregated)
        self.plot_tradeoff_curves(aggregated)
        self.plot_key_finding(aggregated)

        if stats_df is not None:
            plot_figure8_significance_heatmap(stats_df, self.output_dir)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':

    # ── Edit this base path to match your machine ─────────────────────────────
    _BASE = r'D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_bc'
    _NASH = os.path.join(_BASE, 'nash_selection', 'nash_optimal_params.json')
    # ─────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description='Generate all sensitivity figures for BookCrossing'
    )
    parser.add_argument(
        '--aggregated', type=str,
        default=os.path.join(_BASE, 'enhanced_analysis', 'aggregated_results.json'),
    )
    parser.add_argument(
        '--stats-csv', type=str,
        default=os.path.join(_BASE, 'enhanced_analysis',
                             'statistics', 'statistical_tests.csv'),
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_BASE, 'enhanced_analysis', 'plots'),
    )
    parser.add_argument(
        '--nash-json', type=str, default=_NASH,
        help='Path to nash_optimal_params.json (auto-detected from default path)',
    )

    args = parser.parse_args()

    # Load Nash-optimal lambdas into FIXED_LAMBDAS before any figure is drawn
    FIXED_LAMBDAS.update(load_fixed_lambdas(args.nash_json))
    if os.path.exists(args.nash_json):
        print(f'  Loaded Nash λ* values: {FIXED_LAMBDAS}')
    else:
        print(f'  [WARN] nash_optimal_params.json not found — using defaults: {FIXED_LAMBDAS}')
        print(f'         Run nash_parameter_selection_bc.py first for correct λ* values.')

    # The generate_all_figures function is not defined in this module,
    # so we instantiate the plotter and call its methods.
    plotter = SensitivityPlotter(args.output_dir, k=K, nash_json_path=args.nash_json)

    print(f'\nLoading aggregated results from {args.aggregated}...')
    with open(args.aggregated) as f:
        agg = json.load(f)

    # Convert lists to numpy arrays
    for algo in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
        for a_str in agg['alphas']:
            for metric in agg[algo][a_str]:
                for stat in ['mean', 'std', 'min', 'max']:
                    v = agg[algo][a_str][metric].get(stat)
                    if isinstance(v, list):
                        agg[algo][a_str][metric][stat] = np.array(v)

    stats_df = pd.read_csv(args.stats_csv) if os.path.exists(args.stats_csv) else None

    print(f'\nGenerating figures → {args.output_dir}')
    print('─' * 50)

    plotter.plot_all(agg, stats_df=stats_df)

    print('\n' + '─' * 50)
    print(f'Output directory: {args.output_dir}')