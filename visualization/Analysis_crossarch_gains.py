"""
Analysis G — Cross-Architecture Absolute Gains
================================================
Complete 3 × 5 × 2 table (models × alphas × datasets) showing:
  ΔAPLT  = APLT_method − APLT_baseline  (absolute fairness gain)
  ΔNDCG% = (NDCG_baseline − NDCG_method) / NDCG_baseline × 100

Answers RQ3 completely:
  Which architecture benefits most from each method?
  Does higher model accuracy (LightGCN) come at the cost of
  lower mitigation headroom?

Figure: heatmap of ΔAPLT across (model × α) for each method,
        both datasets side by side.

Reads from all_results_combined.csv and aggregated_results.json.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        10.5,
    'axes.titlesize':   11,
    'axes.labelsize':   10.5,
    'xtick.labelsize':  9.5,
    'ytick.labelsize':  9.5,
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

METHOD_LABELS = {
    'BinaryxQuAD': 'Binary xQuAD',
    'SmoothxQuAD': 'Smooth xQuAD',
    'FAIR':        'FA*IR',
}
DS_LABELS = {
    'MovieLens':   'MovieLens-1M',
    'AmazonBooks': 'Amazon Books',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def get_idx(grid, target):
    rounded = [round(g, 4) for g in grid]
    try:    return rounded.index(round(target, 4))
    except: return int(np.argmin(np.abs(np.array(grid) - target)))


def load_bpr(json_path, dataset):
    with open(json_path) as f:
        raw = json.load(f)
    lambda_grid = raw['lambda_grid']
    tau_grid    = raw.get('tau_grid', [round(x * 0.1, 1) for x in range(11)])
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
# COMPUTE ABSOLUTE GAINS
# ─────────────────────────────────────────────────────────────
def compute_gains(df):
    agg = df.groupby(['model', 'dataset', 'alpha', 'method']).agg(
        ndcg=('NDCG@10', 'mean'),
        aplt=('APLT@10', 'mean'),
    ).reset_index()

    baseline = agg[agg['method'] == 'Baseline'][
        ['model', 'dataset', 'alpha', 'ndcg', 'aplt']
    ].rename(columns={'ndcg': 'ndcg_base', 'aplt': 'aplt_base'})

    merged = agg[agg['method'] != 'Baseline'].merge(
        baseline, on=['model', 'dataset', 'alpha']
    )
    merged['delta_aplt']    = merged['aplt'] - merged['aplt_base']
    merged['delta_ndcg_pct']= ((merged['ndcg_base'] - merged['ndcg']) /
                                merged['ndcg_base'] * 100)
    return merged


# ─────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────
def print_table(gains):
    for dataset in DATASETS:
        print(f"\n{'='*90}")
        print(f"  {dataset} — Absolute Gains (ΔAPLT | ΔNDCG%)")
        print(f"{'='*90}")
        header = f"  {'α':>5}  {'Model':>10}"
        for method in METHODS:
            header += f"  {METHOD_LABELS[method]:>22}"
        print(header)
        sub_h = f"  {'':>5}  {'':>10}"
        for _ in METHODS:
            sub_h += f"  {'ΔAPLT':>10}  {'ΔNDCG%':>10}"
        print(sub_h)
        print('  ' + '-'*88)
        sub = gains[gains['dataset'] == dataset]
        for alpha in ALPHA_LEVELS:
            for model in MODELS:
                tag = f"  {alpha:>5.2f}" if model == MODELS[0] else f"  {'':>5}"
                row = f"{tag}  {model:>10}"
                for method in METHODS:
                    v = sub[(sub['model']  == model) &
                            (sub['alpha']  == alpha) &
                            (sub['method'] == method)]
                    if len(v):
                        row += (f"  {v['delta_aplt'].values[0]:>10.4f}"
                                f"  {v['delta_ndcg_pct'].values[0]:>9.2f}%")
                    else:
                        row += f"  {'n/a':>10}  {'n/a':>10}"
                print(row)
            print()


# ─────────────────────────────────────────────────────────────
# LATEX TABLE  — one per dataset
# ─────────────────────────────────────────────────────────────
def write_latex(gains, dataset, outpath):
    ds    = DS_LABELS[dataset]
    label = 'ml' if 'MovieLens' in dataset else 'ab'
    sub   = gains[gains['dataset'] == dataset]

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(
        r'\caption{' + ds +
        r': absolute APLT gain ($\Delta$APLT) and relative NDCG cost '
        r'($\Delta$NDCG\%) at Nash-selected parameters across all three '
        r'model architectures and bias levels. '
        r'LightGCN consistently shows the largest $\Delta$NDCG cost, '
        r'reflecting its more accuracy-focused candidate pool.}'
    )
    lines.append(r'\label{tab:crossarch_gains_' + label + r'}')
    lines.append(r'\begin{tabular}{clcccccc}')
    lines.append(r'\toprule')
    lines.append(
        r'\multirow{2}{*}{$\alpha$} & \multirow{2}{*}{Model} & '
        r'\multicolumn{2}{c}{Binary xQuAD} & '
        r'\multicolumn{2}{c}{Smooth xQuAD} & '
        r'\multicolumn{2}{c}{FA*IR} \\'
    )
    lines.append(
        r' & & $\Delta$APLT & $\Delta$NDCG\% & '
        r'$\Delta$APLT & $\Delta$NDCG\% & '
        r'$\Delta$APLT & $\Delta$NDCG\% \\'
    )
    lines.append(r'\midrule')

    for alpha in ALPHA_LEVELS:
        lines.append(r'\multirow{3}{*}{' + f"{alpha:.2f}" + r'}')
        for i, model in enumerate(MODELS):
            shade = r' \rowcolor{gray!6}' if i % 2 == 1 else ''
            row   = f"{shade} & {model}"
            for method in METHODS:
                v = sub[(sub['model']  == model) &
                        (sub['alpha']  == alpha) &
                        (sub['method'] == method)]
                if len(v):
                    da = v['delta_aplt'].values[0]
                    dn = v['delta_ndcg_pct'].values[0]
                    row += f" & {da:.4f} & {dn:.2f}\\%"
                else:
                    row += r" & n/a & n/a"
            lines.append(row + r' \\')
        lines.append(r'\midrule')

    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


# ─────────────────────────────────────────────────────────────
# FIGURE — heatmap: ΔAPLT across model × α for each method
# ─────────────────────────────────────────────────────────────
def plot_heatmap(gains, save_path):
    """
    3 rows (methods) × 2 columns (datasets).
    Each cell: heatmap with rows=models, cols=alpha levels.
    Colour = ΔAPLT. Annotated with ΔAPLT value.
    Single shared colorbar.
    """
    # Custom green colormap (white → deep green)
    cmap = LinearSegmentedColormap.from_list(
        'aplt_gain', ['#ffffff', '#c7e9c0', '#41ab5d', '#005a32']
    )

    fig, axes = plt.subplots(
        3, 2, figsize=(11, 8),
        gridspec_kw={'hspace': 0.45, 'wspace': 0.25}
    )
    fig.subplots_adjust(left=0.10, right=0.88, top=0.93, bottom=0.08)

    # Global ΔAPLT range for shared colorbar
    vmin = 0.0
    vmax = gains['delta_aplt'].max() * 1.05

    alpha_labels = [str(a) for a in ALPHA_LEVELS]
    model_labels = MODELS

    ims = []
    for row_i, method in enumerate(METHODS):
        for col_i, dataset in enumerate(DATASETS):
            ax  = axes[row_i, col_i]
            sub = gains[(gains['method']  == method) &
                        (gains['dataset'] == dataset)]

            # Build matrix: rows=models, cols=alphas
            matrix = np.zeros((len(MODELS), len(ALPHA_LEVELS)))
            for m_i, model in enumerate(MODELS):
                for a_i, alpha in enumerate(ALPHA_LEVELS):
                    v = sub[(sub['model'] == model) &
                            (sub['alpha'] == alpha)]['delta_aplt']
                    matrix[m_i, a_i] = v.values[0] if len(v) else 0.0

            im = ax.imshow(matrix, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           aspect='auto')
            ims.append(im)

            # Annotate cells
            for m_i in range(len(MODELS)):
                for a_i in range(len(ALPHA_LEVELS)):
                    val     = matrix[m_i, a_i]
                    txt_clr = 'white' if val > vmax * 0.6 else '#222222'
                    ax.text(a_i, m_i, f'{val:.3f}',
                            ha='center', va='center',
                            fontsize=8.5, color=txt_clr, fontweight='bold')

            ax.set_xticks(range(len(ALPHA_LEVELS)))
            ax.set_xticklabels(alpha_labels, fontsize=9)
            ax.set_yticks(range(len(MODELS)))
            ax.set_yticklabels(model_labels if col_i == 0 else [],
                               fontsize=9.5)

            # Titles
            if row_i == 0:
                ax.set_title(DS_LABELS[dataset],
                             fontweight='bold', pad=7, fontsize=10.5)
            if col_i == 0:
                ax.set_ylabel(METHOD_LABELS[method],
                              fontweight='bold', labelpad=8, fontsize=10)
            if row_i == len(METHODS) - 1:
                ax.set_xlabel('Bias level α', labelpad=5)

    # Shared colourbar
    cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.80])
    sm      = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('ΔAPLT (absolute gain)', fontsize=10, labelpad=8)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Analysis G — Cross-Architecture Absolute Gains")
    print("=" * 60)

    df    = load_all()
    gains = compute_gains(df)

    print_table(gains)

    for dataset in DATASETS:
        label = 'ml' if 'MovieLens' in dataset else 'ab'
        write_latex(gains, dataset,
                    os.path.join(OUTPUT_DIR,
                                 f'tab_crossarch_gains_{label}.tex'))

    plot_heatmap(gains,
                 os.path.join(OUTPUT_DIR, 'fig_crossarch_gains.pdf'))
    plot_heatmap(gains,
                 os.path.join(OUTPUT_DIR, 'fig_crossarch_gains.png'))
    print("\nDone.")


if __name__ == "__main__":
    main()