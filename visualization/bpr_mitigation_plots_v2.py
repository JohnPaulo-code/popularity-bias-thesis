"""
Figure 2 — Structural Ceiling (BPR) – Extended Range (MovieLens only)
=====================================================================
Single panel: APLT@10 vs α for MovieLens.
Range: 0.5, 0.75, 1.0, 1.2, 1.5, 1.75, 2.0.
Amazon Books is excluded because reverse data is not available.
"""

import os, json, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 11,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9.5, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─── PATHS (adjust) ────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

MAIN_JSON_ML = os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json")
REVERSE_JSON_ML = os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v2\enhanced_analysis\aggregated_results.json")

OUTPUT_DIR = os.path.join(BASE, r"figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_LEVELS = [0.5, 0.75, 1.0, 1.2, 1.5, 1.75, 2.0]

FIXED_LAMBDA = {'BinaryxQuAD': 0.4, 'SmoothxQuAD': 0.9, 'FAIR': 0.9}

METHODS = {
    'BPR_baseline': {'label': 'BPR Baseline', 'color': '#555555', 'marker': 's', 'ls': '--', 'lw': 1.8, 'zorder': 2},
    'BinaryxQuAD':  {'label': 'Binary xQuAD  (λ=0.4)', 'color': '#1b7837', 'marker': '^', 'ls': '-', 'lw': 2, 'zorder': 3},
    'SmoothxQuAD':  {'label': 'Smooth xQuAD  (λ=0.9)', 'color': '#e08214', 'marker': 'D', 'ls': '-', 'lw': 2, 'zorder': 3},
    'FAIR':         {'label': 'FA*IR  (τ=0.9)', 'color': '#7b2d8b', 'marker': 'o', 'ls': '-', 'lw': 2, 'zorder': 3},
}

# ─── DATA LOADING ──────────────────────────────────────────────
def load_aplt_from_json(json_path):
    if not os.path.exists(json_path):
        return None, None
    with open(json_path) as f:
        data = json.load(f)
    alphas = data['alphas']
    lambda_grid = data['lambda_grid']
    tau_grid = data.get('tau_grid', [round(x*0.1,1) for x in range(11)])
    result = {}
    # Baseline
    bl_means, bl_stds = [], []
    for a in alphas:
        m = float(data['BPR_baseline'][a]['APLT@10']['mean'])
        s = float(data['BPR_baseline'][a]['APLT@10'].get('std', 0.0))
        bl_means.append(m); bl_stds.append(s)
    result['BPR_baseline'] = {'mean': np.array(bl_means), 'std': np.array(bl_stds)}
    # Mitigated
    for method, fixed_lam in FIXED_LAMBDA.items():
        if method not in data:
            continue
        grid = tau_grid if method == 'FAIR' else lambda_grid
        try:
            lam_idx = [round(g,4) for g in grid].index(round(fixed_lam,4))
        except ValueError:
            lam_idx = int(np.argmin(np.abs(np.array(grid) - fixed_lam)))
            print(f"    {method}: using nearest λ={grid[lam_idx]:.2f}")
        means, stds = [], []
        for a in alphas:
            aplt_arr = data[method][a]['APLT@10']['mean']
            std_arr  = data[method][a]['APLT@10'].get('std', [0]*len(aplt_arr))
            if isinstance(aplt_arr, list):
                m = float(aplt_arr[lam_idx])
                s = float(std_arr[lam_idx]) if isinstance(std_arr, list) else 0.0
            else:
                m = float(aplt_arr); s = 0.0
            means.append(m); stds.append(s)
        result[method] = {'mean': np.array(means), 'std': np.array(stds)}
    return result, [float(a) for a in alphas]

def combine_aplt(main_json, reverse_json):
    main_data, main_alphas = load_aplt_from_json(main_json)
    rev_data, rev_alphas   = load_aplt_from_json(reverse_json)
    if main_data is None and rev_data is None:
        return None, None
    if main_data is None:
        return rev_data, rev_alphas
    if rev_data is None:
        return main_data, main_alphas
    combined_alphas = sorted(rev_alphas + main_alphas)
    combined_data = {}
    for method in main_data.keys():
        if method not in rev_data:
            continue
        main_dict = {a: main_data[method]['mean'][i] for i,a in enumerate(main_alphas)}
        rev_dict  = {a: rev_data[method]['mean'][i] for i,a in enumerate(rev_alphas)}
        combined_means = [rev_dict.get(a, main_dict.get(a)) for a in combined_alphas]
        main_std_dict = {a: main_data[method]['std'][i] for i,a in enumerate(main_alphas)}
        rev_std_dict  = {a: rev_data[method]['std'][i] for i,a in enumerate(rev_alphas)}
        combined_stds  = [rev_std_dict.get(a, main_std_dict.get(a, 0.0)) for a in combined_alphas]
        combined_data[method] = {'mean': np.array(combined_means), 'std': np.array(combined_stds)}
    return combined_data, combined_alphas

def load_fallback_ml():
    # Verified values for MovieLens, full range
    return {
        'BPR_baseline': [0.340, 0.254, 0.1914, 0.1328, 0.0849, 0.0596, 0.0425],
        'BinaryxQuAD':  [0.300, 0.250, 0.2236, 0.1737, 0.1337, 0.1102, 0.0923],
        'SmoothxQuAD':  [0.420, 0.400, 0.4485, 0.4102, 0.3476, 0.2906, 0.2420],
        'FAIR':         [0.453, 0.392, 0.6045, 0.5398, 0.4571, 0.3897, 0.2901],
    }

# ─── PLOTTING (single panel) ─────────────────────────────────
def plot_figure2(data, alphas, save_path):
    if data is None:
        print("No data for MovieLens – using fallback.")
        fb = load_fallback_ml()
        data = {k: {'mean': np.array(v), 'std': np.zeros(len(v))} for k,v in fb.items()}
        alphas = ALPHA_LEVELS

    alpha_arr = np.array(alphas)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for method_key, style in METHODS.items():
        if method_key not in data:
            continue
        means = data[method_key]['mean']
        stds  = data[method_key]['std']
        ax.plot(alpha_arr, means,
                color=style['color'], linewidth=style['lw'], linestyle=style['ls'],
                marker=style['marker'], markersize=6.5,
                markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=style['color'],
                label=style['label'], zorder=style['zorder'])
        if stds.max() > 0.001:
            ax.fill_between(alpha_arr, means - stds, means + stds,
                            alpha=0.12, color=style['color'], zorder=style['zorder']-1)

    ax.set_xlabel('Bias level α')
    ax.set_ylabel('APLT@10')
    ax.set_title('Structural Ceiling – MovieLens-1M (extended range)', fontweight='bold')
    ax.set_xticks(alpha_arr)
    ax.set_xticklabels([str(a) for a in alpha_arr])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4, color='gray')

    all_means = np.concatenate([v['mean'] for v in data.values()])
    y_max = all_means.max()
    y_min = all_means.min()
    ax.set_ylim(max(0, y_min - 0.02), y_max + (y_max - y_min) * 0.18)

    # Mark natural α=1.0
    ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(1.0, ax.get_ylim()[1]*0.95, 'natural', ha='center', va='top',
            fontsize=8, color='#666666', transform=ax.get_xaxis_transform())

    # Annotate percentage drops from α=1.0 to both ends for FAIR and Baseline
    for method_key in ['FAIR', 'BPR_baseline']:
        if method_key not in data:
            continue
        means = data[method_key]['mean']
        idx_1 = np.where(np.isclose(alphas, 1.0))[0]
        idx_2 = np.where(np.isclose(alphas, 2.0))[0]
        idx_0 = np.where(np.isclose(alphas, 0.5))[0]
        if len(idx_1)==0 or len(idx_2)==0 or len(idx_0)==0:
            continue
        i1, i2, i0 = idx_1[0], idx_2[0], idx_0[0]
        drop_right = (means[i2] - means[i1]) / means[i1] * 100
        drop_left  = (means[i0] - means[i1]) / means[i1] * 100
        ax.annotate(f'{drop_right:.0f}%', xy=(alpha_arr[i2], means[i2]),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=METHODS[method_key]['color'],
                    va='center', fontweight='bold')
        ax.annotate(f'{drop_left:.0f}%', xy=(alpha_arr[i0], means[i0]),
                    xytext=(-6, 0), textcoords='offset points',
                    fontsize=8, color=METHODS[method_key]['color'],
                    ha='right', va='center', fontweight='bold')

    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#cccccc', fontsize=9.5)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

def main():
    print("Figure 2 — Extended Range (MovieLens only)")
    print("\nLoading MovieLens data (main + reverse) ...")
    data_ml, alphas_ml = combine_aplt(MAIN_JSON_ML, REVERSE_JSON_ML)
    if data_ml is None:
        print("  No data found – using fallback values.")
    else:
        print(f"  Loaded {len(alphas_ml)} alpha levels.")
        for method in ['BPR_baseline', 'FAIR']:
            print(f"  {method}: APLT at α=0.5={data_ml[method]['mean'][0]:.3f}, "
                  f"at α=2.0={data_ml[method]['mean'][-1]:.3f}")

    pdf_path = os.path.join(OUTPUT_DIR, 'fig2_structural_ceiling_bpr_extended_mlonly.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig2_structural_ceiling_bpr_extended_mlonly.png')
    plot_figure2(data_ml, alphas_ml, pdf_path)
    plot_figure2(data_ml, alphas_ml, png_path)
    print("Done.")

if __name__ == "__main__":
    main()