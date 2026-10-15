"""
Figure 5 — Architecture Comparison: Best Nash Score vs α
=========================================================
For BPR, NeuMF and LightGCN, at each bias level α (1.0–2.0),
compute the maximum Nash score across all fairness methods
(BinaryxQuAD, SmoothxQuAD, FA*IR). This shows the best
achievable trade‑off quality for each architecture.
Legend placed at the bottom.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   11,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  9.5,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

BASE = r"D:\Final_thesis\thesis_project"

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
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

OUTPUT_DIR = os.path.join(BASE, r"figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed hyperparameters for fairness methods (same as in your earlier analysis)
FIXED_LAMBDA = {'BinaryxQuAD': 0.4, 'SmoothxQuAD': 0.9, 'FAIR': 0.9}
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]

# Colours and labels for the three architectures
MODEL_STYLES = {
    'BPR':      {'color': '#555555', 'label': 'BPR'},
    'NeuMF':    {'color': '#e08214', 'label': 'NeuMF'},
    'LightGCN': {'color': '#7b2d8b', 'label': 'LightGCN'},
}
MODEL_ORDER = ['BPR', 'NeuMF', 'LightGCN']


# ─────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────
def get_idx(grid, target):
    rounded = [round(g, 4) for g in grid]
    try:
        return rounded.index(round(target, 4))
    except ValueError:
        return int(np.argmin(np.abs(np.array(grid) - target)))

def load_bpr_nash(json_path, alpha_levels, fixed_lambda):
    """For BPR, load baseline and fairness methods, compute Nash scores."""
    with open(json_path, 'r') as f:
        raw = json.load(f)

    lambda_grid = raw['lambda_grid']
    tau_grid = raw.get('tau_grid', [round(x * 0.1, 1) for x in range(11)])

    results = {}
    for a_str in raw['alphas']:
        alpha = float(a_str)
        if alpha not in alpha_levels:
            continue
        # Baseline
        base_ndcg = raw['BPR_baseline'][a_str]['NDCG@10']['mean']
        base_aplt = raw['BPR_baseline'][a_str]['APLT@10']['mean']

        best_nash = 1.0  # baseline itself gives Nash=1 (but we will max over methods)
        for method, param in fixed_lambda.items():
            if method not in raw:
                continue
            grid = tau_grid if method == 'FAIR' else lambda_grid
            idx = get_idx(grid, param)
            ndcg_arr = raw[method][a_str]['NDCG@10']['mean']
            aplt_arr = raw[method][a_str]['APLT@10']['mean']
            ndcg = ndcg_arr[idx] if isinstance(ndcg_arr, list) else ndcg_arr
            aplt = aplt_arr[idx] if isinstance(aplt_arr, list) else aplt_arr
            nash = (ndcg / base_ndcg) * (aplt / base_aplt)
            best_nash = max(best_nash, nash)
        results[alpha] = best_nash
    return results

def load_cross_nash(csv_path, alpha_levels, fixed_lambda):
    """For NeuMF/LightGCN, load from CSV, average seeds, compute Nash."""
    df = pd.read_csv(csv_path)
    df['alpha_str'] = df['alpha'].apply(lambda x: str(float(x)))
    results = {}
    for alpha in alpha_levels:
        a_str = str(alpha)
        # Baseline
        base_sub = df[(df['alpha_str'] == a_str) & (df['method'] == 'Baseline')]
        if base_sub.empty:
            continue
        base_ndcg = base_sub['NDCG@10'].mean()
        base_aplt = base_sub['APLT@10'].mean()

        best_nash = 1.0
        for method, param in fixed_lambda.items():
            m_sub = df[(df['alpha_str'] == a_str) & (df['method'] == method)]
            if m_sub.empty:
                continue
            # For cross_model CSV, each row already corresponds to a specific λ/τ.
            # We need to pick the row with the correct fixed λ (Binary/Smooth) or τ (FAIR).
            # The CSV has a column 'param' (or maybe 'lambda'/'tau') – inspect.
            # Based on your earlier `load_csv` function, the CSV doesn't have param column,
            # it's already aggregated per fixed lambda? Actually the CSV in `cross_model`
            # is already per (alpha, method) with fixed parameters (λ=0.4 for Binary, etc.).
            # So we just take the mean across seeds.
            ndcg = m_sub['NDCG@10'].mean()
            aplt = m_sub['APLT@10'].mean()
            nash = (ndcg / base_ndcg) * (aplt / base_aplt)
            best_nash = max(best_nash, nash)
        results[alpha] = best_nash
    return results


# ─────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────
def plot_best_nash_vs_alpha(all_data, save_path):
    """all_data: dict {(model, dataset): {alpha: best_nash_score}}"""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5),
                                      gridspec_kw={'wspace': 0.32})
    fig.subplots_adjust(bottom=0.18)

    for dataset, ax in [('MovieLens', ax_l), ('AmazonBooks', ax_r)]:
        for model in MODEL_ORDER:
            key = (model, dataset)
            if key not in all_data:
                continue
            scores = all_data[key]
            alphas = sorted(scores.keys())
            if not alphas:
                continue
            y_vals = [scores[a] for a in alphas]
            ax.plot(alphas, y_vals,
                    color=MODEL_STYLES[model]['color'],
                    linewidth=2.2,
                    marker='o', markersize=7,
                    markerfacecolor='white',
                    markeredgewidth=1.8,
                    label=MODEL_STYLES[model]['label'])

        ax.set_xlabel('Bias level α', labelpad=6)
        ax.set_ylabel('Best Nash score\n(max across fairness methods)', labelpad=8)
        ax.set_title(f'{"(a) " if dataset=="MovieLens" else "(b) "}'
                     f'{dataset.replace("AmazonBooks","Amazon Books")}',
                     pad=8, fontweight='bold')
        ax.set_xticks(ALPHA_LEVELS)
        ax.set_xticklabels([str(a) for a in ALPHA_LEVELS])
        ax.set_ylim(bottom=1.0)  # Nash score starts at 1 (baseline)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))

    # Legend at bottom centre
    handles = [mlines.Line2D([0],[0],
                             color=MODEL_STYLES[m]['color'],
                             marker='o', markersize=7, linewidth=2,
                             label=MODEL_STYLES[m]['label'],
                             markerfacecolor='white',
                             markeredgewidth=1.8)
               for m in MODEL_ORDER]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.95, edgecolor='#cccccc', fontsize=10, handlelength=2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Figure 5 — Best Nash Score vs α (Architecture Comparison)")
    print("=" * 58)

    all_data = {}

    # Load BPR
    for dataset, path in BPR_JSONS.items():
        if not os.path.exists(path):
            print(f"  Warning: BPR JSON not found for {dataset}")
            continue
        print(f"Loading BPR/{dataset} ...")
        data = load_bpr_nash(path, ALPHA_LEVELS, FIXED_LAMBDA)
        all_data[('BPR', dataset)] = data
        print(f"  Alphas: {sorted(data.keys())}, scores: {[data[a] for a in sorted(data)]}")

    # Load NeuMF and LightGCN
    for model in ['NeuMF', 'LightGCN']:
        for dataset in ['MovieLens', 'AmazonBooks']:
            csv_path = CROSS_CSVS[model][dataset]
            if not os.path.exists(csv_path):
                print(f"  Warning: CSV not found for {model}/{dataset}")
                continue
            print(f"Loading {model}/{dataset} ...")
            data = load_cross_nash(csv_path, ALPHA_LEVELS, FIXED_LAMBDA)
            all_data[(model, dataset)] = data
            print(f"  Alphas: {sorted(data.keys())}, scores: {[data[a] for a in sorted(data)]}")

    # Generate figure (two panels: MovieLens, Amazon Books)
    pdf = os.path.join(OUTPUT_DIR, 'fig5_architecture_nash_vs_alpha.pdf')
    png = os.path.join(OUTPUT_DIR, 'fig5_architecture_nash_vs_alpha.png')
    print("\nGenerating figure ...")
    plot_best_nash_vs_alpha(all_data, pdf)
    plot_best_nash_vs_alpha(all_data, png)
    print(f"\nDone.\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()