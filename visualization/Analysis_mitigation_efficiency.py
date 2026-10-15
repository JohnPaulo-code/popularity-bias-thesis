"""
Analysis D — Mitigation Efficiency
=====================================
Efficiency = APLT gain / NDCG cost (%)

Figure: 2 panels (MovieLens | Amazon Books).
Each panel: 3 lines, one per method.
Each point: efficiency averaged across all 3 models (BPR, NeuMF, LightGCN).

This answers: which mitigation strategy converts NDCG sacrifice
into APLT gain most effectively, regardless of base model?
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

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

# Method colours — consistent with all other figures
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
    tau_grid    = raw.get('tau_grid', [round(x*0.1,1) for x in range(11)])
    rows = []
    for a in raw['alphas']:
        bl_ndcg = float(raw['BPR_baseline'][a]['NDCG@10']['mean'])
        bl_aplt = float(raw['BPR_baseline'][a]['APLT@10']['mean'])
        rows.append({'model':'BPR','dataset':dataset,'alpha':float(a),
                     'method':'Baseline','NDCG@10':bl_ndcg,'APLT@10':bl_aplt})
        for method in METHODS:
            if method not in raw: continue
            grid = tau_grid if method=='FAIR' else lambda_grid
            idx  = get_idx(grid, NASH[dataset][method])
            na   = raw[method][a]['NDCG@10']['mean']
            aa   = raw[method][a]['APLT@10']['mean']
            rows.append({'model':'BPR','dataset':dataset,'alpha':float(a),
                         'method':method,
                         'NDCG@10': float(na[idx] if isinstance(na,list) else na),
                         'APLT@10': float(aa[idx] if isinstance(aa,list) else aa)})
    return pd.DataFrame(rows)


def load_cross(csv_path, model, dataset):
    df = pd.read_csv(csv_path)
    df = df[df['method'].isin(['Baseline'] + METHODS)].copy()
    df['model']   = model
    df['dataset'] = dataset
    df['alpha']   = df['alpha'].astype(float)
    return df[['model','dataset','alpha','method','NDCG@10','APLT@10']]


def load_all():
    frames = []
    for dataset in DATASETS:
        frames.append(load_bpr(BPR_JSONS[dataset], dataset))
        for model in ['NeuMF','LightGCN']:
            path = CROSS_CSVS[model][dataset]
            if os.path.exists(path):
                frames.append(load_cross(path, model, dataset))
            else:
                print(f"  ⚠ Not found: {path}")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# COMPUTE EFFICIENCY — averaged across models
# ─────────────────────────────────────────────────────────────
def compute_efficiency(df):
    """
    Per (model, dataset, alpha, method): compute efficiency.
    Then average across models to get one value per
    (dataset, alpha, method).
    """
    agg = df.groupby(['model','dataset','alpha','method']).agg(
        ndcg=('NDCG@10','mean'),
        aplt=('APLT@10','mean')
    ).reset_index()

    baseline = agg[agg['method']=='Baseline'][
        ['model','dataset','alpha','ndcg','aplt']
    ].rename(columns={'ndcg':'ndcg_base','aplt':'aplt_base'})

    merged = agg[agg['method']!='Baseline'].merge(
        baseline, on=['model','dataset','alpha']
    )
    merged['aplt_gain']     = merged['aplt'] - merged['aplt_base']
    merged['ndcg_cost_pct'] = ((merged['ndcg_base'] - merged['ndcg']) /
                                merged['ndcg_base'] * 100)
    merged['efficiency']    = np.where(
        merged['ndcg_cost_pct'] > 0.001,
        merged['aplt_gain'] / merged['ndcg_cost_pct'],
        np.nan
    )

    # Average across the three models
    avg = merged.groupby(['dataset','alpha','method']).agg(
        efficiency_mean=('efficiency','mean'),
        efficiency_std =('efficiency','std'),
        aplt_gain_mean =('aplt_gain','mean'),
        ndcg_cost_mean =('ndcg_cost_pct','mean'),
    ).reset_index()

    return merged, avg


# ─────────────────────────────────────────────────────────────
# PRINT TABLES
# ─────────────────────────────────────────────────────────────
def print_tables(merged, avg):
    print("\n  Per-model efficiency:")
    for dataset in DATASETS:
        print(f"\n  {dataset}")
        sub = merged[merged['dataset']==dataset]
        print(f"  {'α':>5}  {'Method':>14}  {'BPR':>8}  {'NeuMF':>8}  {'LightGCN':>10}  {'Mean':>8}")
        print('  ' + '-'*66)
        for alpha in ALPHA_LEVELS:
            for method in METHODS:
                row = f"  {alpha:>5.2f}  {method:>14}" if method==METHODS[0] else f"  {'':>5}  {method:>14}"
                for model in MODELS:
                    v = sub[(sub['model']==model)&(sub['alpha']==alpha)&
                            (sub['method']==method)]['efficiency']
                    row += f"  {v.values[0]:>8.4f}" if len(v) else f"  {'n/a':>8}"
                # add mean
                m = avg[(avg['dataset']==dataset)&(avg['alpha']==alpha)&
                        (avg['method']==method)]['efficiency_mean']
                row += f"  {m.values[0]:>8.4f}" if len(m) else ""
                print(row)
            print()


# ─────────────────────────────────────────────────────────────
# LATEX TABLE — average efficiency
# ─────────────────────────────────────────────────────────────
def write_latex(avg, outpath):
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Mitigation efficiency (APLT gain per 1\% NDCG cost) '
        r'averaged across BPR, NeuMF, and LightGCN. '
        r'Higher values indicate a more favourable accuracy--fairness trade-off. '
        r'Smooth xQuAD consistently achieves the highest efficiency, '
        r'while Binary xQuAD efficiency is the most stable across bias levels.}'
    )
    lines.append(r'\label{tab:efficiency_avg}')
    lines.append(r'\begin{tabular}{clcc}')
    lines.append(r'\toprule')
    lines.append(
        r'$\alpha$ & Method & MovieLens-1M & Amazon Books \\'
    )
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
                v = avg[(avg['dataset']==dataset)&(avg['alpha']==alpha)&
                        (avg['method']==method)]['efficiency_mean']
                vals.append(f"{v.values[0]:.4f}" if len(v) else 'n/a')
            lines.append(
                f"{shade}{a_str} & {mname} & {vals[0]} & {vals[1]} \\\\"
            )
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outpath,'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


# ─────────────────────────────────────────────────────────────
# FIGURE — 2 panels, 3 lines each (average across models)
# ─────────────────────────────────────────────────────────────
def plot_efficiency(avg, save_path):
    alpha_arr = np.array(ALPHA_LEVELS)

    ds_labels = {
        'MovieLens': '(a) MovieLens-1M',
        'AmazonBooks': '(b) Amazon Books'
    }

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(10, 4.8),
        gridspec_kw={'wspace': 0.35}
    )

    # Make room for legend + note
    fig.subplots_adjust(bottom=0.28)

    axes = {
        'MovieLens': ax_l,
        'AmazonBooks': ax_r
    }

    for dataset, ax in axes.items():
        sub = avg[avg['dataset'] == dataset].sort_values('alpha')

        for method in METHODS:
            s = METHOD_STYLES[method]

            m = sub[sub['method'] == method].sort_values('alpha')
            if m.empty:
                continue

            eff_vals = m['efficiency_mean'].values
            eff_std = m['efficiency_std'].values

            ax.plot(
                alpha_arr,
                eff_vals,
                color=s['color'],
                linewidth=s['lw'],
                marker=s['marker'],
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.8,
                label=s['label'],
                zorder=3
            )

            # ±1 std band across models
            ax.fill_between(
                alpha_arr,
                eff_vals - eff_std,
                eff_vals + eff_std,
                color=s['color'],
                alpha=0.12,
                zorder=2
            )

        ax.set_xlabel('Bias level α', labelpad=6)
        ax.set_title(
            ds_labels[dataset],
            pad=8,
            fontweight='bold'
        )

        ax.set_xticks(alpha_arr)
        ax.set_xticklabels([str(a) for a in ALPHA_LEVELS])

        ax.set_ylim(bottom=0)

        ax.grid(
            True,
            linestyle='--',
            linewidth=0.5,
            alpha=0.4,
            color='gray'
        )

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.3f}')
        )

    ax_l.set_ylabel(
        'Efficiency (APLT gain per 1% NDCG cost)',
        labelpad=6
    )

    # ---------------------------------------------------------
    # Shared legend BELOW both panels
    # ---------------------------------------------------------
    handles, labels = ax_r.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=10,
        columnspacing=1.8,
        handlelength=2.2
    )

    # ---------------------------------------------------------
    # Shared note BELOW legend
    # ---------------------------------------------------------
    fig.text(
        0.5,
        0.02,
        'Efficiency averaged across BPR, NeuMF and LightGCN. '
        'Shaded bands show ±1 std across architectures.',
        ha='center',
        fontsize=9,
        color='#555555'
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"  Figure → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  Analysis D — Mitigation Efficiency (avg across models)")
    print("="*60)

    df           = load_all()
    merged, avg  = compute_efficiency(df)
    print_tables(merged, avg)

    write_latex(avg, os.path.join(OUTPUT_DIR, 'tab_efficiency_avg.tex'))

    plot_efficiency(avg, os.path.join(OUTPUT_DIR, 'fig_efficiency.pdf'))
    plot_efficiency(avg, os.path.join(OUTPUT_DIR, 'fig_efficiency.png'))
    print("\nDone.")


if __name__ == "__main__":
    main()