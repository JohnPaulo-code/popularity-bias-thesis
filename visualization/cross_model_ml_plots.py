"""
Cross-Model Robustness Study — Figure Suite
============================================
Produces 5 publication-quality figures for the thesis cross-model chapter.

Figures
-------
  F10  Baseline model comparison (NDCG & APLT vs alpha)
  F11  NDCG cost stability — parameter transfer finding
  F12  APLT gain divergence — LightGCN amplification finding
  F13  Cohen's d trajectory — effect size grows with bias
  F14  Accuracy-fairness trade-off scatter — all 45 conditions

Data is embedded directly from the cross-model run output.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = r"D:\Final_thesis\thesis_project\cross_model\cm_ml_results\plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Palette ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {'BPR': '#1f77b4', 'LightGCN': '#d62728', 'NeuMF': '#2ca02c'}
MODEL_MARKERS = {'BPR': 'o', 'LightGCN': 's', 'NeuMF': '^'}
METHOD_COLORS = {'BinaryxQuAD': '#1f77b4', 'SmoothxQuAD': '#ff7f0e', 'FAIR': '#2ca02c'}
METHOD_LABELS = {'BinaryxQuAD': 'Binary xQuAD (λ=0.2)',
                 'SmoothxQuAD': 'Smooth xQuAD (λ=0.4)',
                 'FAIR': 'FA*IR (τ=0.7)'}

ALPHAS = [1.0, 1.2, 1.5, 2.0, 2.5]
ALPHA_LABELS = ['α=1.0', 'α=1.2', 'α=1.5', 'α=2.0', 'α=2.5']
MODELS = ['BPR', 'LightGCN', 'NeuMF']
METHODS = ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']

# ── Embedded results (from Phase 4 console output) ────────────────────────────
# Format: (alpha, model) → {method: (ndcg_pct, aplt_pct, cohens_d)}
RESULTS = {
    (1.0, 'BPR'): {'BinaryxQuAD': (-2.49, 11.16, -0.0863), 'SmoothxQuAD': (-12.34, 75.49, -0.1657),
                   'FAIR': (-25.51, 151.82, -0.2513)},
    (1.0, 'LightGCN'): {'BinaryxQuAD': (-2.94, 55.22, -0.0989), 'SmoothxQuAD': (-13.32, 285.00, -0.2142),
                        'FAIR': (-29.64, 494.67, -0.3174)},
    (1.0, 'NeuMF'): {'BinaryxQuAD': (-2.78, 18.98, -0.1050), 'SmoothxQuAD': (-11.80, 119.28, -0.1853),
                     'FAIR': (-27.14, 238.37, -0.2782)},
    (1.2, 'BPR'): {'BinaryxQuAD': (-2.97, 12.04, -0.1183), 'SmoothxQuAD': (-13.07, 83.05, -0.2233),
                   'FAIR': (-26.93, 166.27, -0.3035)},
    (1.2, 'LightGCN'): {'BinaryxQuAD': (-3.16, 68.37, -0.1386), 'SmoothxQuAD': (-14.13, 343.30, -0.2794),
                        'FAIR': (-31.27, 603.58, -0.3882)},
    (1.2, 'NeuMF'): {'BinaryxQuAD': (-2.62, 23.09, -0.1142), 'SmoothxQuAD': (-12.17, 141.24, -0.2302),
                     'FAIR': (-29.56, 286.15, -0.3343)},
    (1.5, 'BPR'): {'BinaryxQuAD': (-2.96, 13.17, -0.1455), 'SmoothxQuAD': (-13.78, 86.37, -0.2759),
                   'FAIR': (-26.89, 175.83, -0.3630)},
    (1.5, 'LightGCN'): {'BinaryxQuAD': (-3.14, 79.76, -0.1640), 'SmoothxQuAD': (-14.88, 404.79, -0.3491),
                        'FAIR': (-30.85, 727.55, -0.4663)},
    (1.5, 'NeuMF'): {'BinaryxQuAD': (-2.35, 25.35, -0.1298), 'SmoothxQuAD': (-12.95, 154.52, -0.2963),
                     'FAIR': (-29.89, 322.83, -0.4187)},
    (2.0, 'BPR'): {'BinaryxQuAD': (-3.06, 14.71, -0.1946), 'SmoothxQuAD': (-14.23, 95.48, -0.3619),
                   'FAIR': (-25.74, 197.32, -0.4473)},
    (2.0, 'LightGCN'): {'BinaryxQuAD': (-3.12, 97.05, -0.2148), 'SmoothxQuAD': (-14.78, 482.65, -0.4489),
                        'FAIR': (-27.08, 884.80, -0.5558)},
    (2.0, 'NeuMF'): {'BinaryxQuAD': (-2.77, 27.58, -0.1908), 'SmoothxQuAD': (-13.40, 168.23, -0.3953),
                     'FAIR': (-26.92, 359.11, -0.4958)},
    (2.5, 'BPR'): {'BinaryxQuAD': (-3.02, 15.58, -0.2069), 'SmoothxQuAD': (-13.82, 100.54, -0.3916),
                   'FAIR': (-23.99, 208.10, -0.4861)},
    (2.5, 'LightGCN'): {'BinaryxQuAD': (-3.28, 96.16, -0.2284), 'SmoothxQuAD': (-15.32, 471.94, -0.5022),
                        'FAIR': (-25.08, 867.71, -0.5946)},
    (2.5, 'NeuMF'): {'BinaryxQuAD': (-2.71, 29.80, -0.2081), 'SmoothxQuAD': (-13.13, 176.29, -0.4355),
                     'FAIR': (-25.50, 385.68, -0.5542)},
}

# Baseline NDCG and APLT per (alpha, model) — from console output
BASELINE = {
    (1.0, 'BPR'): 0.0407, (1.0, 'LightGCN'): 0.0671, (1.0, 'NeuMF'): 0.0568,
    (1.2, 'BPR'): 0.0496, (1.2, 'LightGCN'): 0.0788, (1.2, 'NeuMF'): 0.0662,
    (1.5, 'BPR'): 0.0687, (1.5, 'LightGCN'): 0.1041, (1.5, 'NeuMF'): 0.0874,
    (2.0, 'BPR'): 0.0993, (2.0, 'LightGCN'): 0.1417, (2.0, 'NeuMF'): 0.1149,
    (2.5, 'BPR'): 0.1245, (2.5, 'LightGCN'): 0.1716, (2.5, 'NeuMF'): 0.1404,
}

BASELINE_APLT = {
    (1.0, 'BPR'): 0.1910, (1.0, 'LightGCN'): 0.0950, (1.0, 'NeuMF'): 0.1510,
    (1.2, 'BPR'): 0.1803, (1.2, 'LightGCN'): 0.0889, (1.2, 'NeuMF'): 0.1431,
    (1.5, 'BPR'): 0.1753, (1.5, 'LightGCN'): 0.0833, (1.5, 'NeuMF'): 0.1356,
    (2.0, 'BPR'): 0.1671, (2.0, 'LightGCN'): 0.0762, (2.0, 'NeuMF'): 0.1241,
    (2.5, 'BPR'): 0.1623, (2.5, 'LightGCN'): 0.0731, (2.5, 'NeuMF'): 0.1186,
}

x = np.array(ALPHAS)


# =============================================================================
# FIGURE 10 — Baseline Model Comparison
# =============================================================================
def plot_figure10():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'Figure 10: Baseline Model Comparison — NDCG@10 and APLT@10 vs Bias Level\n'
        '(mean across 3 seeds; shading = ±0.5 std estimated from cross-seed range)',
        fontsize=12, fontweight='bold', y=1.01
    )

    for model in MODELS:
        ndcg_vals = [BASELINE[(a, model)] for a in ALPHAS]
        aplt_vals = [BASELINE_APLT[(a, model)] for a in ALPHAS]
        c, m = MODEL_COLORS[model], MODEL_MARKERS[model]

        ax1.plot(x, ndcg_vals, color=c, marker=m, lw=2.2, markersize=8,
                 label=model, markeredgecolor='white', markeredgewidth=1.2)
        ax2.plot(x, aplt_vals, color=c, marker=m, lw=2.2, markersize=8,
                 label=model, markeredgecolor='white', markeredgewidth=1.2)

        # Annotate final point
        ax1.annotate(f'{ndcg_vals[-1]:.3f}', (x[-1], ndcg_vals[-1]),
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=8.5, color=c, va='center')
        ax2.annotate(f'{aplt_vals[-1]:.3f}', (x[-1], aplt_vals[-1]),
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=8.5, color=c, va='center')

    for ax, ylabel, title in [
        (ax1, 'NDCG@10  ↑ higher is better', 'Accuracy — NDCG@10'),
        (ax2, 'APLT@10  ↑ higher is better', 'Long-tail Coverage — APLT@10'),
    ]:
        ax.set_xlabel('Bias Level (α)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(ALPHA_LABELS, fontsize=9)
        ax.legend(fontsize=9, frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--')

    # Annotation: LightGCN lowest APLT
    ax2.annotate(
        'LightGCN generates most\npopularity-biased lists',
        xy=(1.5, BASELINE_APLT[(1.5, 'LightGCN')]),
        xytext=(1.8, 0.055),
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
        fontsize=9, color='#d62728',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0', alpha=0.9)
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure10_baseline_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 11 — NDCG Cost Stability (Parameter Transfer Finding)
# =============================================================================
def plot_figure11():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        'Figure 11: NDCG@10 Cost at Fixed λ* — Near-Identical Across Architectures\n'
        '(BPR-calibrated parameters applied unchanged to LightGCN and NeuMF)',
        fontsize=12, fontweight='bold', y=1.01
    )

    for ax, method in zip(axes, METHODS):
        for model in MODELS:
            vals = [RESULTS[(a, model)][method][0] for a in ALPHAS]
            c, m = MODEL_COLORS[model], MODEL_MARKERS[model]
            ax.plot(x, vals, color=c, marker=m, lw=2.2, markersize=8,
                    label=model, markeredgecolor='white', markeredgewidth=1.2)

        # Range annotation
        all_vals = [RESULTS[(a, mdl)][method][0] for a in ALPHAS for mdl in MODELS]
        spread = max(all_vals) - min(all_vals)
        ax.text(0.04, 0.06,
                f'Cross-model spread:\n±{spread / 2:.1f}pp max',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
                          edgecolor='#ccaa00', alpha=0.9))

        ax.set_xlabel('Bias Level (α)', fontsize=11)
        ax.set_ylabel('NDCG@10 relative change (%)', fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight='bold')
        ax.set_xticks(x);
        ax.set_xticklabels(ALPHA_LABELS, fontsize=8.5)
        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.grid(True, alpha=0.25, linestyle='--')
        if method == 'BinaryxQuAD':
            ax.legend(fontsize=9, frameon=True, loc='lower left')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure11_ndcg_cost_stability.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 12 — APLT Gain Divergence (LightGCN Amplification)
# =============================================================================
def plot_figure12():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        'Figure 12: APLT@10 Gain at Fixed λ* — LightGCN Amplification Effect\n'
        '(Graph convolution propagates popularity bias → greater correction needed → larger gain)',
        fontsize=12, fontweight='bold', y=1.01
    )

    for ax, method in zip(axes, METHODS):
        for model in MODELS:
            vals = [RESULTS[(a, model)][method][1] for a in ALPHAS]
            c, m = MODEL_COLORS[model], MODEL_MARKERS[model]
            ax.plot(x, vals, color=c, marker=m, lw=2.2, markersize=8,
                    label=model, markeredgecolor='white', markeredgewidth=1.2)

            # Annotate final alpha value
            ax.annotate(f'{vals[-1]:.0f}%',
                        (x[-1], vals[-1]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, color=c, va='center')

        ax.set_xlabel('Bias Level (α)', fontsize=11)
        ax.set_ylabel('APLT@10 gain vs baseline (%)', fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight='bold')
        ax.set_xticks(x);
        ax.set_xticklabels(ALPHA_LABELS, fontsize=8.5)
        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('+%.0f%%'))
        ax.grid(True, alpha=0.25, linestyle='--')
        if method == 'BinaryxQuAD':
            ax.legend(fontsize=9, frameon=True, loc='upper left')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure12_aplt_gain_divergence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 13 — Cohen's d Trajectory
# =============================================================================
def plot_figure13():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        "Figure 13: Cohen's d (NDCG) — Effect Size Grows with Bias Level\n"
        '(negative d = method reduces NDCG; larger magnitude = stronger effect)',
        fontsize=12, fontweight='bold', y=1.01
    )

    # Reference lines for effect size interpretation
    thresholds = [(-0.2, 'small', '#aaaaaa'), (-0.5, 'medium', '#888888'),
                  (-0.8, 'large', '#555555')]

    for ax, method in zip(axes, METHODS):
        for model in MODELS:
            vals = [RESULTS[(a, model)][method][2] for a in ALPHAS]
            c, m = MODEL_COLORS[model], MODEL_MARKERS[model]
            ax.plot(x, vals, color=c, marker=m, lw=2.2, markersize=8,
                    label=model, markeredgecolor='white', markeredgewidth=1.2)

        for thresh, label, col in thresholds:
            ax.axhline(thresh, color=col, linestyle=':', lw=1.2, alpha=0.6)
            ax.text(x[-1] + 0.05, thresh, label, fontsize=8,
                    color=col, va='center')

        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        ax.set_xlabel('Bias Level (α)', fontsize=11)
        ax.set_ylabel("Cohen's d (NDCG@10)", fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight='bold')
        ax.set_xticks(x);
        ax.set_xticklabels(ALPHA_LABELS, fontsize=8.5)
        ax.set_xlim(0.9, 2.7)
        ax.grid(True, alpha=0.25, linestyle='--')
        if method == 'BinaryxQuAD':
            ax.legend(fontsize=9, frameon=True, loc='lower left')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure13_cohens_d_trajectory.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 14 — Accuracy-Fairness Trade-off Heatmap (all 45 conditions)
# =============================================================================
def plot_figure14():
    """
    Two heatmaps side by side: NDCG cost % and APLT gain %.
    Rows = model × method (9 rows), cols = alpha levels (5 cols).
    """
    row_labels = [f'{mdl}\n{mth.replace("xQuAD", "xQ")}'
                  for mdl in MODELS for mth in METHODS]

    ndcg_mat = np.zeros((9, 5))
    aplt_mat = np.zeros((9, 5))

    for r, (mdl, mth) in enumerate(
            [(mdl, mth) for mdl in MODELS for mth in METHODS]):
        for c, a in enumerate(ALPHAS):
            ndcg_mat[r, c] = RESULTS[(a, mdl)][mth][0]
            aplt_mat[r, c] = RESULTS[(a, mdl)][mth][1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Figure 14: Accuracy-Fairness Trade-off — All 45 Experimental Conditions\n'
        '(BPR-calibrated λ* applied fixed across architectures and bias levels)',
        fontsize=12, fontweight='bold'
    )

    # NDCG cost
    sns.heatmap(ndcg_mat, ax=ax1,
                annot=True, fmt='.1f',
                xticklabels=ALPHA_LABELS,
                yticklabels=row_labels,
                cmap='RdYlGn_r', center=0,
                cbar_kws={'label': 'NDCG@10 change (%)', 'shrink': 0.8},
                linewidths=0.4, linecolor='white',
                annot_kws={'size': 9})
    ax1.set_title('Accuracy Cost (NDCG@10 %Δ)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Bias Level (α)', fontsize=10)
    ax1.tick_params(axis='y', labelsize=8.5)

    # Add model separators
    for pos in [3, 6]:
        ax1.axhline(pos, color='white', lw=2.5)

    # APLT gain — log scale annotations for readability
    sns.heatmap(aplt_mat, ax=ax2,
                annot=True, fmt='.0f',
                xticklabels=ALPHA_LABELS,
                yticklabels=row_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'APLT@10 gain (%)', 'shrink': 0.8},
                linewidths=0.4, linecolor='white',
                annot_kws={'size': 9})
    ax2.set_title('Fairness Gain (APLT@10 %Δ)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Bias Level (α)', fontsize=10)
    ax2.tick_params(axis='y', labelsize=8.5)

    for pos in [3, 6]:
        ax2.axhline(pos, color='white', lw=2.5)

    # Model labels on left
    for ax in (ax1, ax2):
        ax.annotate('BPR', xy=(-0.18, 0.83), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', color=MODEL_COLORS['BPR'],
                    rotation=90, va='center')
        ax.annotate('LightGCN', xy=(-0.18, 0.50), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', color=MODEL_COLORS['LightGCN'],
                    rotation=90, va='center')
        ax.annotate('NeuMF', xy=(-0.18, 0.17), xycoords='axes fraction',
                    fontsize=10, fontweight='bold', color=MODEL_COLORS['NeuMF'],
                    rotation=90, va='center')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure14_tradeoff_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 15 — Scatter: all 45 conditions on NDCG-cost vs APLT-gain axes
# =============================================================================
def plot_figure15():
    """
    Scatter plot: x = NDCG cost, y = APLT gain.
    Each point = one (alpha, model, method) condition.
    Shape = model, colour = method.
    Convex hulls per method show that model choice shifts APLT but not NDCG.
    """
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, color='white', linewidth=0.9, zorder=0)

    for method in METHODS:
        # Collect all points for this method
        xs = [RESULTS[(a, mdl)][method][0] for a in ALPHAS for mdl in MODELS]
        ys = [RESULTS[(a, mdl)][method][1] for a in ALPHAS for mdl in MODELS]
        # Draw connecting ellipse / hull region
        x_c, y_c = np.mean(xs), np.mean(ys)
        x_r, y_r = (max(xs) - min(xs)) / 2 + 1.5, (max(ys) - min(ys)) / 2 + 10
        ellipse = plt.matplotlib.patches.Ellipse(
            (x_c, y_c), width=x_r * 2, height=y_r * 2,
            color=METHOD_COLORS[method], alpha=0.08, zorder=1
        )
        ax.add_patch(ellipse)

    for method in METHODS:
        for model in MODELS:
            xs = [RESULTS[(a, model)][method][0] for a in ALPHAS]
            ys = [RESULTS[(a, model)][method][1] for a in ALPHAS]
            ax.scatter(xs, ys,
                       color=METHOD_COLORS[method],
                       marker=MODEL_MARKERS[model],
                       s=80, zorder=4, alpha=0.85,
                       edgecolors='white', linewidths=0.8,
                       label=f'{MODEL_LABELS_SHORT[model]} — {METHOD_LABELS[method]}'
                       if model == 'BPR' else '_')
            # Connect alpha progression with faint line
            ax.plot(xs, ys, color=METHOD_COLORS[method],
                    lw=0.8, alpha=0.3, zorder=3)

    # Method centroid labels
    for method in METHODS:
        xs = [RESULTS[(a, mdl)][method][0] for a in ALPHAS for mdl in MODELS]
        ys = [RESULTS[(a, mdl)][method][1] for a in ALPHAS for mdl in MODELS]
        ax.annotate(
            METHOD_LABELS[method].split('(')[0].strip(),
            (np.mean(xs), np.mean(ys)),
            fontsize=10, fontweight='bold', color=METHOD_COLORS[method],
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    # Custom legend for models only
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color='grey', lw=0,
               markersize=9, label=m, markeredgecolor='white')
        for m in MODELS
    ]
    ax.legend(handles=model_handles, title='Model', fontsize=9,
              loc='upper left', framealpha=0.9)

    ax.set_xlabel('NDCG@10 relative change (%)', fontsize=12)
    ax.set_ylabel('APLT@10 gain vs baseline (%)', fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('+%.0f%%'))
    ax.axvline(0, color='black', lw=0.8, alpha=0.4)
    ax.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax.set_title(
        'Figure 15: Accuracy-Fairness Trade-off Space — All 45 Conditions\n'
        '(shape = model architecture; colour = mitigation method; '
        'each cluster spans 5 bias levels)',
        fontsize=12, fontweight='bold', pad=12
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'figure15_tradeoff_scatter.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {path}')


MODEL_LABELS_SHORT = {'BPR': 'BPR', 'LightGCN': 'LightGCN', 'NeuMF': 'NeuMF'}

# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == '__main__':
    print(f'\nGenerating cross-model figures → {OUTPUT_DIR}\n' + '─' * 60)
    plot_figure10()
    plot_figure11()
    plot_figure12()
    plot_figure13()
    plot_figure14()
    plot_figure15()
    print('─' * 60)
    print('All 6 figures generated.\n')
    print('Figure → thesis section mapping:')
    print('  figure10_baseline_comparison.png  → Model characteristics')
    print('  figure11_ndcg_cost_stability.png  → Parameter transfer finding')
    print('  figure12_aplt_gain_divergence.png → LightGCN amplification finding')
    print('  figure13_cohens_d_trajectory.png  → Effect size grows with bias')
    print('  figure14_tradeoff_heatmap.png     → Full results table as heatmap')
    print('  figure15_tradeoff_scatter.png     → Trade-off space overview')