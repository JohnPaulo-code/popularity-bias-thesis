"""
Analysis F — Relative APLT Improvement
========================================
Relative improvement (%) = (APLT_method - APLT_baseline) / APLT_baseline × 100

This reveals the Binary xQuAD paradox:
  - Binary xQuAD  : relative gain INCREASES with α
                    (baseline collapses faster than Binary's absolute gain)
  - FA*IR         : relative gain DECREASES with α
                    (structural ceiling shrinks faster than baseline)
  - Smooth xQuAD  : intermediate, depends on dataset

Figure: 2 panels (MovieLens | Amazon Books)
        3 lines per panel (one per method), averaged across models
        Shaded band = ±1 std across BPR / NeuMF / LightGCN

Table:  relative improvement at each α, averaged across models

Reads directly from:
  aggregated_results.json  (BPR)
  all_results_combined.csv (NeuMF, LightGCN)
No reranking re-run needed.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   11,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

BPR_JSONS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json"),
    'AmazonBooks': os.path.join(BASE, r"sensitivity_analysis\sa_results_ab_v1\enhanced_analysis\aggregated_results.json"),
}
CROSS_CSVS = {
    'NeuMF': {
        'MovieLens':   os.path.join(BASE, r"cross_model\NeuMF\MovieLens\all_results_combined.csv"),
        'AmazonBooks': os.path.join(BASE, r"cross_model\NeuMF\AmazonBooks\all_results_combined.csv"),
    },
    'LightGCN': {
        'MovieLens':   os.path.join(BASE, r"cross_model\LightGCN\MovieLens\all_results_combined.csv"),
        'AmazonBooks': os.path.join(BASE, r"cross_model\LightGCN\AmazonBooks\all_results_combined.csv"),
    },
}

OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
DATASETS     = ['MovieLens', 'AmazonBooks']
MODELS       = ['BPR', 'NeuMF', 'LightGCN']
METHODS      = ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']

NASH = {
    'MovieLens':   {'BinaryxQuAD': 0.4, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
    'AmazonBooks': {'BinaryxQuAD': 0.3, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
}

METHOD_STYLES = {
    'BinaryxQuAD': {'color': '#1b7837', 'marker': '^', 'lw': 2.0,
                    'label': 'Binary xQuAD'},
    'SmoothxQuAD': {'color': '#e08214', 'marker': 'D', 'lw': 2.0,
                    'label': 'Smooth xQuAD'},
    'FAIR':        {'color': '#7b2d8b', 'marker': 'o', 'lw': 2.0,
                    'label': 'FA*IR'},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (same helpers as Analysis D)
# ─────────────────────────────────────────────────────────────
def get_idx(grid, target):
    rounded = [round(g, 4) for g in grid]
    try:    return rounded.index(round(target, 4))
    except: return int(np.argmin(np.abs(np.array(grid) - target)))


def load_bpr(json_path, dataset):
    with open(json_path) as f:
        raw = json.load(f)
    lambda_grid = raw['lambda_grid']
    tau_grid    = raw.get('tau_grid', [round(x*0.1, 1) for x in range(11)])
    rows = []
    for a in raw['alphas']:
        bl_ndcg = float(raw['BPR_baseline'][a]['NDCG@10']['mean'])
        bl_aplt = float(raw['BPR_baseline'][a]['APLT@10']['mean'])
        rows.append({'model': 'BPR', 'dataset': dataset, 'alpha': float(a),
                     'method': 'Baseline',
                     'NDCG@10': bl_ndcg, 'APLT@10': bl_aplt})
        for method in METHODS:
            if method not in raw:
                continue
            grid = tau_grid if method == 'FAIR' else lambda_grid
            idx  = get_idx(grid, NASH[dataset][method])
            na   = raw[method][a]['NDCG@10']['mean']
            aa   = raw[method][a]['APLT@10']['mean']
            rows.append({'model': 'BPR', 'dataset': dataset, 'alpha': float(a),
                         'method': method,
                         'NDCG@10': float(na[idx] if isinstance(na, list) else na),
                         'APLT@10': float(aa[idx] if isinstance(aa, list) else aa)})
    return pd.DataFrame(rows)


def load_cross(csv_path, model, dataset):
    df = pd.read_csv(csv_path)
    df = df[df['method'].isin(['Baseline'] + METHODS)].copy()
    df['model']   = model
    df['dataset'] = dataset
    df['alpha']   = df['alpha'].astype(float)
    return df[['model', 'dataset', 'alpha', 'method', 'NDCG@10', 'APLT@10']]


def load_all():
    frames = []
    for dataset in DATASETS:
        frames.append(load_bpr(BPR_JSONS[dataset], dataset))
        for model in ['NeuMF', 'LightGCN']:
            path = CROSS_CSVS[model][dataset]
            if os.path.exists(path):
                frames.append(load_cross(path, model, dataset))
            else:
                print(f"  ⚠  Not found: {path}")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# COMPUTE RELATIVE IMPROVEMENT
# ─────────────────────────────────────────────────────────────
def compute_relative(df):
    """
    Relative APLT improvement (%) per (model, dataset, alpha, method).
    Then average across models.
    """
    agg = df.groupby(['model', 'dataset', 'alpha', 'method']).agg(
        aplt=('APLT@10', 'mean'),
        ndcg=('NDCG@10', 'mean'),
    ).reset_index()

    baseline = agg[agg['method'] == 'Baseline'][
        ['model', 'dataset', 'alpha', 'aplt', 'ndcg']
    ].rename(columns={'aplt': 'aplt_base', 'ndcg': 'ndcg_base'})

    merged = agg[agg['method'] != 'Baseline'].merge(
        baseline, on=['model', 'dataset', 'alpha']
    )

    merged['rel_aplt_pct'] = np.where(
        merged['aplt_base'] > 0,
        (merged['aplt'] - merged['aplt_base']) / merged['aplt_base'] * 100,
        np.nan
    )
    merged['abs_aplt_gain'] = merged['aplt'] - merged['aplt_base']
    merged['ndcg_cost_pct'] = ((merged['ndcg_base'] - merged['ndcg']) /
                                merged['ndcg_base'] * 100)

    # Average across models
    avg = merged.groupby(['dataset', 'alpha', 'method']).agg(
        rel_mean =('rel_aplt_pct',  'mean'),
        rel_std  =('rel_aplt_pct',  'std'),
        abs_mean =('abs_aplt_gain', 'mean'),
        ndcg_mean=('ndcg_cost_pct', 'mean'),
    ).reset_index()

    return merged, avg


# ─────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────
def print_table(merged, avg):
    for dataset in DATASETS:
        print(f"\n{'='*72}")
        print(f"  {dataset} — Relative APLT Improvement (%)")
        print(f"  (APLT_method − APLT_baseline) / APLT_baseline × 100")
        print(f"{'='*72}")
        print(f"  {'α':>5}  {'Method':>14}  {'BPR':>8}  "
              f"{'NeuMF':>8}  {'LightGCN':>10}  {'Mean':>8}")
        print('  ' + '-'*62)
        for alpha in ALPHA_LEVELS:
            for method in METHODS:
                tag = f"  {alpha:>5.2f}" if method == METHODS[0] else f"  {'':>5}"
                row = f"{tag}  {method:>14}"
                for model in MODELS:
                    v = merged[(merged['model']   == model)   &
                               (merged['dataset'] == dataset) &
                               (merged['alpha']   == alpha)   &
                               (merged['method']  == method)]['rel_aplt_pct']
                    row += f"  {v.values[0]:>8.1f}%" if len(v) else f"  {'n/a':>8}"
                m = avg[(avg['dataset'] == dataset) &
                        (avg['alpha']   == alpha)   &
                        (avg['method']  == method)]['rel_mean']
                row += f"  {m.values[0]:>8.1f}%" if len(m) else ""
                print(row)
            print()


# ─────────────────────────────────────────────────────────────
# LATEX TABLE
# ─────────────────────────────────────────────────────────────
def write_latex(avg, outpath):
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Relative APLT improvement (\%) at Nash-selected parameters, '
        r'averaged across BPR, NeuMF, and LightGCN. '
        r'Binary xQuAD shows a paradoxical increase in relative gain as '
        r'$\alpha$ rises: the baseline APLT collapses faster than Binary '
        r"xQuAD's modest absolute gain, inflating the relative figure. "
        r"FA*IR's relative gain decreases with $\alpha$ as the structural "
        r'ceiling shrinks.}'
    )
    lines.append(r'\label{tab:relative_improvement}')
    lines.append(r'\begin{tabular}{clcc}')
    lines.append(r'\toprule')
    lines.append(r'$\alpha$ & Method & MovieLens-1M (\%) & Amazon Books (\%) \\')
    lines.append(r'\midrule')

    for alpha in ALPHA_LEVELS:
        first = True
        for i, method in enumerate(METHODS):
            a_str = f"{alpha:.2f}" if first else ""
            first = False
            shade = r'\rowcolor{gray!6} ' if i % 2 == 1 else ''
            mname = METHOD_STYLES[method]['label']
            vals  = []
            for dataset in DATASETS:
                v = avg[(avg['dataset'] == dataset) &
                        (avg['alpha']   == alpha)   &
                        (avg['method']  == method)]['rel_mean']
                vals.append(f"{v.values[0]:.1f}" if len(v) else 'n/a')
            lines.append(
                f"{shade}{a_str} & {mname} & {vals[0]}\\% & {vals[1]}\\% \\\\"
            )
        lines.append(r'\midrule')

    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


# ─────────────────────────────────────────────────────────────
# FIGURE — 2 panels, 3 lines, average across models
# ─────────────────────────────────────────────────────────────
def plot_relative(avg, save_path):
    alpha_arr = np.array(ALPHA_LEVELS)
    ds_labels = {'MovieLens':   '(a) MovieLens-1M',
                 'AmazonBooks': '(b) Amazon Books'}

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(10, 4.2),
        gridspec_kw={'wspace': 0.35}
    )
    axes = {'MovieLens': ax_l, 'AmazonBooks': ax_r}

    for dataset, ax in axes.items():
        sub = avg[avg['dataset'] == dataset]

        for method in METHODS:
            s = METHOD_STYLES[method]
            m = sub[sub['method'] == method].sort_values('alpha')
            if m.empty:
                continue
            vals = m['rel_mean'].values
            stds = m['rel_std'].values

            ax.plot(alpha_arr, vals,
                    color=s['color'], linewidth=s['lw'],
                    marker=s['marker'], markersize=7,
                    markerfacecolor='white', markeredgewidth=1.8,
                    label=s['label'], zorder=3)

            ax.fill_between(alpha_arr,
                            vals - stds, vals + stds,
                            alpha=0.12, color=s['color'], zorder=2)

        ax.set_xlabel('Bias level α', labelpad=6)
        ax.set_title(ds_labels[dataset], pad=8, fontweight='bold')
        ax.set_xticks(alpha_arr)
        ax.set_xticklabels([str(a) for a in ALPHA_LEVELS])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.0f}%')
        )
        ax.grid(True, linestyle='--', linewidth=0.5,
                alpha=0.4, color='gray')

    ax_l.set_ylabel('Relative APLT improvement (%)', labelpad=6)
    ax_r.legend(loc='upper right', framealpha=0.92,
                edgecolor='#cccccc', fontsize=10)

    fig.text(
        0.5, -0.04,
        'Relative improvement averaged across BPR, NeuMF and LightGCN. '
        'Shaded bands show ±1 std across architectures.',
        ha='center', fontsize=9, color='#555555'
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Analysis F — Relative APLT Improvement")
    print("=" * 60)

    df          = load_all()
    merged, avg = compute_relative(df)

    print_table(merged, avg)
    write_latex(avg, os.path.join(OUTPUT_DIR, 'tab_relative_improvement.tex'))

    plot_relative(avg, os.path.join(OUTPUT_DIR, 'fig_relative_improvement.pdf'))
    plot_relative(avg, os.path.join(OUTPUT_DIR, 'fig_relative_improvement.png'))
    print("\nDone.")


if __name__ == "__main__":
    main()