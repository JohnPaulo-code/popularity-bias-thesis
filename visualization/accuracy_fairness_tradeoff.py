"""
Figure 4 — Cross-Model Consistency: Method Trajectories across α
===============================================================
Supports MODEL = 'BPR', 'NeuMF', or 'LightGCN'.
Each line connects the same method for α = 1.0 → 1.2 → 1.5 → 1.75 → 2.0.
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

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — choose model here
# ─────────────────────────────────────────────────────────────
MODEL = 'BPR'          # 'BPR', 'NeuMF', or 'LightGCN'

BASE = r"D:\Final_thesis\thesis_project"

CONFIGS = {
    'BPR': {
        'ml_json': os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json"),
        'ab_json': os.path.join(BASE, r"sensitivity_analysis\sa_results_ab_v1\enhanced_analysis\aggregated_results.json"),
        'out_prefix': 'fig4c_pareto_bpr',
        'title_suffix': 'BPR',
        'fixed_lambda': {'BinaryxQuAD': 0.4, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
    },
    'NeuMF': {
        'ml_csv': os.path.join(BASE, r"cross_model\NeuMF\MovieLens\all_results_combined.csv"),
        'ab_csv': os.path.join(BASE, r"cross_model\NeuMF\AmazonBooks\all_results_combined.csv"),
        'out_prefix': 'fig4a_pareto_neumf',
        'title_suffix': 'NeuMF',
    },
    'LightGCN': {
        'ml_csv': os.path.join(BASE, r"cross_model\LightGCN\MovieLens\all_results_combined.csv"),
        'ab_csv': os.path.join(BASE, r"cross_model\LightGCN\AmazonBooks\all_results_combined.csv"),
        'out_prefix': 'fig4b_pareto_lightgcn',
        'title_suffix': 'LightGCN',
    },
}

OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = ['1.0', '1.2', '1.5', '1.75', '2.0']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per‑method colours (same for all models)
METHOD_COLORS = {
    'Baseline':    '#1b9e77',
    'BinaryxQuAD': '#d95f02',
    'SmoothxQuAD': '#7570b3',
    'FAIR':        '#e7298a',
}
METHOD_ORDER = ['Baseline', 'BinaryxQuAD', 'SmoothxQuAD', 'FAIR']
METHOD_LABELS = {
    'Baseline':    'Baseline',
    'BinaryxQuAD': 'Binary\nxQuAD',
    'SmoothxQuAD': 'Smooth\nxQuAD',
    'FAIR':        'FA*IR',
}
METHOD_MARKERS = {
    'Baseline':    's',
    'BinaryxQuAD': '^',
    'SmoothxQuAD': 'D',
    'FAIR':        'o',
}
METHOD_MS = {
    'Baseline':    52,
    'BinaryxQuAD': 54,
    'SmoothxQuAD': 50,
    'FAIR':        55,
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_csv(csv_path):
    """Load CSV, average seeds, return dict: {alpha_str: {method: {'ndcg','aplt'}}}"""
    df = pd.read_csv(csv_path)
    df['alpha_str'] = df['alpha'].apply(lambda x: str(float(x)))
    result = {}
    for alpha_str in ALPHA_LEVELS:
        sub = df[df['alpha_str'] == alpha_str]
        if sub.empty:
            continue
        result[alpha_str] = {}
        for method in METHOD_ORDER:
            m_sub = sub[sub['method'] == method]
            if m_sub.empty:
                continue
            result[alpha_str][method] = {
                'ndcg': float(m_sub['NDCG@10'].mean()),
                'aplt': float(m_sub['APLT@10'].mean()),
            }
    return result, ALPHA_LEVELS


def load_bpr_json(json_path, fixed_lambda):
    """
    Load BPR JSON exactly as in Figure 3, then convert to standard format.
    Returns (data, alphas) where data[alpha][method] = {'ndcg', 'aplt'}.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    alphas = raw['alphas']
    lambda_grid = raw['lambda_grid']
    tau_grid = raw.get('tau_grid', [round(x * 0.1, 1) for x in range(11)])

    def get_idx(grid, target):
        rounded = [round(g, 4) for g in grid]
        try:
            return rounded.index(round(target, 4))
        except ValueError:
            return int(np.argmin(np.abs(np.array(grid) - target)))

    data = {}
    for a in alphas:
        if a not in raw['BPR_baseline']:
            continue
        row = {}
        # Baseline
        row['Baseline'] = {
            'ndcg': float(raw['BPR_baseline'][a]['NDCG@10']['mean']),
            'aplt': float(raw['BPR_baseline'][a]['APLT@10']['mean']),
        }
        # Fairness methods
        for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
            if method not in raw:
                continue
            grid = tau_grid if method == 'FAIR' else lambda_grid
            idx = get_idx(grid, fixed_lambda[method])
            ndcg_vals = raw[method][a]['NDCG@10']['mean']
            aplt_vals = raw[method][a]['APLT@10']['mean']
            ndcg = ndcg_vals[idx] if isinstance(ndcg_vals, list) else ndcg_vals
            aplt = aplt_vals[idx] if isinstance(aplt_vals, list) else aplt_vals
            row[method] = {'ndcg': float(ndcg), 'aplt': float(aplt)}
        data[a] = row

    # Keep only alphas that exist in data and are in ALPHA_LEVELS
    alphas_present = [a for a in ALPHA_LEVELS if a in data]
    return data, alphas_present


# ─────────────────────────────────────────────────────────────
# PANEL: method trajectories across α (method labels removed)
# ─────────────────────────────────────────────────────────────
def draw_panel(ax, data, alphas, dataset_name, panel_label):
    all_ndcg = []
    for a in alphas:
        if a not in data:
            continue
        for m in METHOD_ORDER:
            if m in data[a]:
                all_ndcg.append(data[a][m]['ndcg'])

    # Draw trajectory lines (methods are identified by legend)
    for method in METHOD_ORDER:
        xs, ys = [], []
        for a in alphas:
            if a in data and method in data[a]:
                xs.append(data[a][method]['ndcg'])
                ys.append(data[a][method]['aplt'])
        if len(xs) > 1:
            ax.plot(xs, ys,
                    color=METHOD_COLORS[method],
                    linewidth=1.8, linestyle='-',
                    marker=METHOD_MARKERS[method],
                    markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                    zorder=3, label=METHOD_LABELS[method].replace('\n', ' '))

    # Annotate α values on FA*IR points (kept for orientation)
    for a in alphas:
        if a not in data or 'FAIR' not in data[a]:
            continue
        pt = data[a]['FAIR']
        idx = alphas.index(a)
        v_offset = 6 if idx % 2 == 0 else -10
        ax.annotate(f'α={a}', xy=(pt['ndcg'], pt['aplt']),
                    xytext=(5, v_offset), textcoords='offset points',
                    fontsize=8, color='black', va='center', ha='left')

    # Method labels have been removed – legend suffices

    ax.set_xlabel('NDCG@10', labelpad=6)
    ax.set_ylabel('APLT@10', labelpad=6)
    ax.set_title(f'{panel_label} {dataset_name}', pad=8, fontweight='bold')
    ax.set_ylim(bottom=0)
    if all_ndcg:
        ax.set_xlim(left=min(all_ndcg) * 0.88, right=max(all_ndcg) * 1.02)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='gray')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.3f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.2f}'))


# ─────────────────────────────────────────────────────────────
# FULL FIGURE
# ─────────────────────────────────────────────────────────────
def plot_figure(data_ml, alphas_ml, data_ab, alphas_ab, model_name, save_path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5),
                                      gridspec_kw={'wspace': 0.38})
    fig.suptitle(f'Accuracy–Fairness Trade‑off across α — {model_name}',
                 fontsize=12, fontweight='bold', y=1.01)

    draw_panel(ax_l, data_ml, alphas_ml, 'MovieLens-1M', '(a)')
    draw_panel(ax_r, data_ab, alphas_ab, 'Amazon Books',  '(b)')

    # Create legend from method handles
    method_handles = []
    for m in METHOD_ORDER:
        h = mlines.Line2D([0], [0], color=METHOD_COLORS[m],
                          marker=METHOD_MARKERS[m], markersize=7, linewidth=1.8,
                          label=METHOD_LABELS[m].replace('\n', ' '),
                          markeredgecolor='white', markeredgewidth=1.0)
        method_handles.append(h)

    fig.legend(handles=method_handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.10), framealpha=0.9, edgecolor='#cccccc',
               fontsize=9, handlelength=2.0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    if MODEL not in CONFIGS:
        raise ValueError(f"Unknown MODEL '{MODEL}'. Choose from {list(CONFIGS.keys())}")

    cfg = CONFIGS[MODEL]
    print("=" * 58)
    print(f"  Figure 4 — Method Trajectories ({MODEL})")
    print("=" * 58)

    if MODEL == 'BPR':
        # Load BPR JSON
        print(f"\nLoading MovieLens ({MODEL}) from JSON...")
        if not os.path.exists(cfg['ml_json']):
            raise FileNotFoundError(f"JSON file not found: {cfg['ml_json']}")
        data_ml, alphas_ml = load_bpr_json(cfg['ml_json'], cfg['fixed_lambda'])
        print(f"  Loaded {len(data_ml)} alpha levels")

        print(f"\nLoading AmazonBooks ({MODEL}) from JSON...")
        if not os.path.exists(cfg['ab_json']):
            raise FileNotFoundError(f"JSON file not found: {cfg['ab_json']}")
        data_ab, alphas_ab = load_bpr_json(cfg['ab_json'], cfg['fixed_lambda'])
        print(f"  Loaded {len(data_ab)} alpha levels")

    else:
        # NeuMF or LightGCN: load CSV
        print(f"\nLoading MovieLens ({MODEL}) from CSV...")
        if not os.path.exists(cfg['ml_csv']):
            raise FileNotFoundError(f"CSV file not found: {cfg['ml_csv']}")
        data_ml, alphas_ml = load_csv(cfg['ml_csv'])
        print(f"  Loaded {len(data_ml)} alpha levels")

        print(f"\nLoading AmazonBooks ({MODEL}) from CSV...")
        if not os.path.exists(cfg['ab_csv']):
            raise FileNotFoundError(f"CSV file not found: {cfg['ab_csv']}")
        data_ab, alphas_ab = load_csv(cfg['ab_csv'])
        print(f"  Loaded {len(data_ab)} alpha levels")

    # Print summary
    for dataset_name, data, alphas in [
        ('MovieLens', data_ml, alphas_ml),
        ('AmazonBooks', data_ab, alphas_ab),
    ]:
        print(f"\n  {dataset_name}:")
        for a in alphas:
            if a not in data:
                continue
            r = data[a]
            bl   = r.get('Baseline', {})
            fair = r.get('FAIR', {})
            print(f"    α={a}: Baseline ({bl.get('ndcg',0):.4f}, {bl.get('aplt',0):.4f})  "
                  f"FA*IR ({fair.get('ndcg',0):.4f}, {fair.get('aplt',0):.4f})")

    pdf = os.path.join(OUTPUT_DIR, f"{cfg['out_prefix']}.pdf")
    png = os.path.join(OUTPUT_DIR, f"{cfg['out_prefix']}.png")

    print("\nGenerating figure ...")
    plot_figure(data_ml, alphas_ml, data_ab, alphas_ab, cfg['title_suffix'], pdf)
    plot_figure(data_ml, alphas_ml, data_ab, alphas_ab, cfg['title_suffix'], png)

    print(f"\nDone.\n  {pdf}\n  {png}")


if __name__ == "__main__":
    main()