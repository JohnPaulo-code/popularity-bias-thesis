"""
Nash Parameter Selection — Table and Figure  (CORRECTED)
=========================================================
Changes vs the original (two fixes, both addressing the same wrong claim
that the Nash score "declines" / "deteriorates" with bias — it RISES,
because the baseline APLT denominator collapses):

  1. plot_nash_vs_alpha(): the in-figure footnote no longer says
     "Declining curves show the best achievable trade-off quality
     deteriorates as bias increases." It now states that the rising
     curves reflect relative headroom over a collapsing baseline.

  2. write_latex(): the table caption no longer claims the score's
     "decline with alpha quantifies the structural ceiling." It now
     matches the corrected tab_nash_selection_ml/ab.tex wording
     (rising = inflation; parameters stable up to plateau equivalence).

Nothing else is changed. The console print_table still shows Unicode
λ/τ, which is terminal-only and harmless; the LaTeX path uses
\\lambda / \\tau correctly.

For each (dataset, method, α):
  1. Sweep all λ / τ values
  2. Compute Nash score = (NDCG(λ)/NDCG_base) × (APLT(λ)/APLT_base)
  3. Record the optimal λ*/τ* and its Nash score
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   10.5,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

BPR_JSONS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json"),
    'AmazonBooks': os.path.join(BASE, r"sensitivity_analysis\sa_results_ab_v1\enhanced_analysis\aggregated_results.json"),
}
OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
METHODS      = ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']

METHOD_LABELS = {
    'BinaryxQuAD': 'Binary xQuAD',
    'SmoothxQuAD': 'Smooth xQuAD',
    'FAIR':        'FA*IR',
}
DS_LABELS = {
    'MovieLens':   'MovieLens-1M',
    'AmazonBooks': 'Amazon Books',
}
PARAM_LABEL = {
    'BinaryxQuAD': 'λ',
    'SmoothxQuAD': 'λ',
    'FAIR':        'τ',
}

# Colour scheme — red (α=1.0) → purple (α=2.0)
ALPHA_COLORS = {
    1.0:  '#d73027',
    1.2:  '#fc8d59',
    1.5:  '#4dac26',
    1.75: '#4575b4',
    2.0:  '#762a83',
}

# Method colours for the companion figure
METHOD_COLORS = {
    'BinaryxQuAD': '#1b7837',
    'SmoothxQuAD': '#e08214',
    'FAIR':        '#7b2d8b',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING AND NASH COMPUTATION
# ─────────────────────────────────────────────────────────────
def load_nash_data(json_path):
    """
    Returns {method: {alpha: {'grid': [...], 'scores': [...],
                               'best_param': X, 'best_score': Y}}}
    """
    with open(json_path) as f:
        raw = json.load(f)

    alphas      = [float(a) for a in raw['alphas']]
    lambda_grid = raw['lambda_grid']
    tau_grid    = raw.get('tau_grid',
                          [round(x * 0.1, 1) for x in range(11)])

    results = {}
    for method in METHODS:
        if method not in raw:
            continue
        grid    = tau_grid if method == 'FAIR' else lambda_grid
        results[method] = {}

        for a_str in raw['alphas']:
            alpha     = float(a_str)
            bl_ndcg   = float(raw['BPR_baseline'][a_str]['NDCG@10']['mean'])
            bl_aplt   = float(raw['BPR_baseline'][a_str]['APLT@10']['mean'])

            ndcg_arr  = np.array(raw[method][a_str]['NDCG@10']['mean'])
            aplt_arr  = np.array(raw[method][a_str]['APLT@10']['mean'])

            # Nash score at each grid point
            nash = (ndcg_arr / max(bl_ndcg, 1e-9)) * \
                   (aplt_arr / max(bl_aplt, 1e-9))

            best_idx   = int(np.argmax(nash))
            results[method][alpha] = {
                'grid':       np.array(grid),
                'nash':       nash,
                'best_param': grid[best_idx],
                'best_score': float(nash[best_idx]),
            }

    return results


# ─────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────
def print_table(dataset_name, data):
    print(f"\n{'='*72}")
    print(f"  {dataset_name} — Nash Selection")
    print(f"  Format: param* (Nash score)")
    print(f"{'='*72}")
    alpha_str = "  ".join(f"{'α='+str(a):>16}" for a in ALPHA_LEVELS)
    print(f"  {'Method':>14}  {alpha_str}")
    print('  ' + '-'*70)
    for method in METHODS:
        if method not in data:
            continue
        row = f"  {METHOD_LABELS[method]:>14}"
        p   = PARAM_LABEL[method]
        for alpha in ALPHA_LEVELS:
            d   = data[method].get(alpha, {})
            val = f"{p}*={d.get('best_param',0):.1f} ({d.get('best_score',0):.3f})"
            row += f"  {val:>16}"
        print(row)


# ─────────────────────────────────────────────────────────────
# LATEX TABLE   (CAPTION FIXED)
# ─────────────────────────────────────────────────────────────
def write_latex(dataset_name, data, outpath):
    ds    = DS_LABELS[dataset_name]
    label = 'ml' if 'MovieLens' in dataset_name else 'ab'
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{' + ds +
        r': Nash-selected parameter ($\lambda^*$ or $\tau^*$) and '
        r'corresponding Nash score (in parentheses) at each bias level '
        r'$\alpha$. The Nash score balances accuracy and fairness '
        r'improvements over the baseline. Scores rise with $\alpha$ '
        r'because the baseline APLT collapses, inflating the relative '
        r'gain (growing headroom); this does not indicate an improving '
        r'absolute trade-off (see Section~\ref{sec:nash_selection}). '
        r'Parameters are stable across $\alpha$ up to plateau '
        r'equivalence, confirming that a single calibration suffices.}'
    )
    lines.append(r'\label{tab:nash_selection_' + label + r'}')

    n_alpha = len(ALPHA_LEVELS)
    col_fmt = 'l' + 'c' * n_alpha
    lines.append(r'\begin{tabular}{' + col_fmt + r'}')
    lines.append(r'\toprule')

    alpha_header = ' & '.join(f'$\\alpha={a}$' for a in ALPHA_LEVELS)
    lines.append(r'Method & ' + alpha_header + r' \\')
    lines.append(r'\midrule')

    for i, method in enumerate(METHODS):
        if method not in data:
            continue
        shade = r'\rowcolor{gray!6} ' if i % 2 == 1 else ''
        p     = PARAM_LABEL[method]                 # console only
        p_tex = r'\tau' if method == 'FAIR' else r'\lambda'   # LaTeX-safe
        row   = f"{shade}{METHOD_LABELS[method]}"
        for alpha in ALPHA_LEVELS:
            d = data[method].get(alpha, {})
            row += (f" & ${p_tex}^*={d.get('best_param',0):.1f}$ "
                    f"({d.get('best_score',0):.3f})")
        lines.append(row + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — Nash score curves (2 rows × 3 cols)
# ─────────────────────────────────────────────────────────────
def plot_nash_curves(all_data, save_path):
    """
    2 rows (datasets) × 3 columns (methods).
    Each subplot: Nash score vs λ/τ, one line per α.
    """
    datasets = ['MovieLens', 'AmazonBooks']
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6.5),
        gridspec_kw={'hspace': 0.42, 'wspace': 0.30}
    )
    fig.subplots_adjust(bottom=0.14, top=0.92)

    import matplotlib.lines as mlines

    for row_i, dataset in enumerate(datasets):
        data = all_data[dataset]
        for col_i, method in enumerate(METHODS):
            ax = axes[row_i, col_i]
            if method not in data:
                ax.text(0.5, 0.5, 'n/a', ha='center',
                        va='center', transform=ax.transAxes)
                continue

            for alpha in ALPHA_LEVELS:
                d = data[method].get(alpha, {})
                if not d:
                    continue
                grid = d['grid']
                nash = d['nash']
                color = ALPHA_COLORS[alpha]
                ax.plot(grid, nash, color=color,
                        linewidth=1.6, alpha=0.8, zorder=3)

            first_alpha = next(iter(data[method].keys())) if data[method] else None
            if first_alpha:
                grid = data[method][first_alpha]['grid']
                ax.set_xlim(grid[0] - 0.02, grid[-1] + 0.02)
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', linewidth=0.4,
                    alpha=0.4, color='gray')

            if row_i == 0:
                ax.set_title(METHOD_LABELS[method],
                             fontweight='bold', pad=6)
            if col_i == 0:
                ax.set_ylabel(
                    f'{DS_LABELS[dataset]}\n\nNash score',
                    labelpad=5, fontsize=9.5
                )
            ax.set_xlabel(
                f'Trade-off parameter '
                f'${PARAM_LABEL[method]}$',
                labelpad=4
            )

    legend_handles = [
        mlines.Line2D([0],[0], color=ALPHA_COLORS[a],
                      linewidth=2, label=f'α={a}')
        for a in ALPHA_LEVELS
    ]
    fig.legend(legend_handles,
               [f'α={a}' for a in ALPHA_LEVELS],
               loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.01),
               framealpha=0.95, edgecolor='#cccccc',
               fontsize=9.5, handlelength=2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure (curves) → {save_path}")


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — Nash score at optimal λ* vs α (2 panels)  (FOOTNOTE FIXED)
# ─────────────────────────────────────────────────────────────
def plot_nash_vs_alpha(all_data, save_path):
    """
    2 panels (ML | AB).
    X axis: α level.  Y axis: Nash score at the optimal λ*/τ* for that α.
    Lines: one per method.
    """
    import matplotlib.lines as mlines

    datasets  = ['MovieLens', 'AmazonBooks']
    ds_labels = {'MovieLens':   '(a) MovieLens-1M',
                 'AmazonBooks': '(b) Amazon Books'}
    alpha_arr = np.array(ALPHA_LEVELS)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(10, 4.5),
        gridspec_kw={'wspace': 0.32}
    )
    axes = {'MovieLens': ax_l, 'AmazonBooks': ax_r}
    fig.subplots_adjust(bottom=0.18)

    for dataset, ax in axes.items():
        data = all_data[dataset]
        for method in METHODS:
            if method not in data:
                continue
            scores = [data[method][a]['best_score']
                      for a in ALPHA_LEVELS
                      if a in data[method]]
            ax.plot(alpha_arr, scores,
                    color=METHOD_COLORS[method],
                    linewidth=2.2,
                    marker='o', markersize=7,
                    markerfacecolor='white',
                    markeredgewidth=1.8,
                    label=METHOD_LABELS[method],
                    zorder=3)

        ax.set_xlabel('Bias level α', labelpad=6)
        ax.set_title(ds_labels[dataset], pad=8, fontweight='bold')
        ax.set_xticks(alpha_arr)
        ax.set_xticklabels([str(a) for a in ALPHA_LEVELS])
        ax.set_ylim(bottom=1.0)
        ax.grid(True, linestyle='--', linewidth=0.5,
                alpha=0.4, color='gray')
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.2f}')
        )

    ax_l.set_ylabel('Nash score at optimal parameter', labelpad=6)

    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.01),
               ncol=3,
               framealpha=0.92,
               edgecolor='#cccccc',
               fontsize=10)

    # FIXED FOOTNOTE — the curves rise; this no longer claims they decline.
    fig.text(
        0.5, -0.04,
        'Nash score = (NDCG($\\lambda^*$) / NDCG$_{\\rm base}$) '
        '$\\times$ (APLT($\\lambda^*$) / APLT$_{\\rm base}$). '
        'Rising curves reflect growing relative headroom over a '
        'collapsing baseline, not improving absolute trade-offs.',
        ha='center', fontsize=8.5, color='#555555'
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure (vs alpha) → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Nash Parameter Selection — Tables and Figures (corrected)")
    print("=" * 60)

    all_data = {}
    for dataset, path in BPR_JSONS.items():
        if not os.path.exists(path):
            print(f"  ⚠  Not found: {path}")
            all_data[dataset] = {}
            continue
        print(f"\nLoading {dataset} ...")
        all_data[dataset] = load_nash_data(path)
        print_table(dataset, all_data[dataset])

        label = 'ml' if 'MovieLens' in dataset else 'ab'
        write_latex(dataset, all_data[dataset],
                    os.path.join(OUTPUT_DIR,
                                 f'tab_nash_selection_{label}.tex'))

    print("\nGenerating Nash score curves figure ...")
    plot_nash_curves(all_data, os.path.join(OUTPUT_DIR, 'fig_nash_curves.pdf'))
    plot_nash_curves(all_data, os.path.join(OUTPUT_DIR, 'fig_nash_curves.png'))

    print("\nGenerating Nash score vs alpha figure ...")
    plot_nash_vs_alpha(all_data, os.path.join(OUTPUT_DIR, 'fig_nash_vs_alpha.pdf'))
    plot_nash_vs_alpha(all_data, os.path.join(OUTPUT_DIR, 'fig_nash_vs_alpha.png'))
    print("\nDone.")


if __name__ == "__main__":
    main()