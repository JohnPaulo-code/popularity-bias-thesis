"""
Figure 2 — The Structural Ceiling (BPR)
=========================================
Two-panel figure: MovieLens (left) and Amazon Books (right).
  X axis : Bias level α
  Y axis : APLT@10
  Lines  : BPR Baseline, BinaryxQuAD (λ=0.4),
            SmoothxQuAD (λ=0.9), FA*IR (τ=0.9)

Story: all four ceilings decline together as α increases —
the structural limit is not method-specific, it is a property
of the candidate pool composition.

Data loaded from stage2 aggregated_results.json.
Fixed parameters match Nash consensus:
  BinaryxQuAD λ=0.4 | SmoothxQuAD λ=0.9 | FA*IR τ=0.9
"""

import os
import json
import numpy as np
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
    'legend.fontsize':  9.5,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

STAGE2_RESULTS = {
    'MovieLens': os.path.join(
        BASE, r"sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json"
    ),
    'AmazonBooks': os.path.join(
        BASE, r"sensitivity_analysis\sa_results_ab_v1\enhanced_analysis\aggregated_results.json"
    ),
}

OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nash consensus parameters
FIXED_LAMBDA = {
    'BinaryxQuAD': 0.4,
    'SmoothxQuAD': 0.9,
    'FAIR':        0.9,
}

# Visual style per method — consistent across all thesis figures
METHODS = {
    'BPR_baseline': {
        'label':  'BPR Baseline',
        'color':  '#555555',
        'marker': 's',
        'ls':     '--',
        'lw':     1.8,
        'zorder': 2,
    },
    'BinaryxQuAD': {
        'label':  'Binary xQuAD  (λ=0.4 / 0.3)',
        'color':  '#1b7837',
        'marker': '^',
        'ls':     '-',
        'lw':     2,
        'zorder': 3,
    },
    'SmoothxQuAD': {
        'label':  'Smooth xQuAD  (λ=0.9)',
        'color':  '#e08214',
        'marker': 'D',
        'ls':     '-',
        'lw':     2,
        'zorder': 3,
    },
    'FAIR': {
        'label':  'FA*IR  (τ=0.9)',
        'color':  '#7b2d8b',
        'marker': 'o',
        'ls':     '-',
        'lw':     2,
        'zorder': 3,
    },
}

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_aplt(json_path, dataset_name):
    """
    Load APLT@10 mean ± std per method at the fixed Nash parameter.

    Returns dict:
      {method_key: {'mean': array[5], 'std': array[5]}}
    """
    with open(json_path) as f:
        data = json.load(f)

    alphas      = data['alphas']           # list of str e.g. ['1.0', ...]
    lambda_grid = data['lambda_grid']      # list of floats
    tau_grid    = data.get('tau_grid',
                           [round(x * 0.1, 1) for x in range(11)])

    result = {}

    # ── Baseline ──────────────────────────────────────────────
    bl_means, bl_stds = [], []
    for a in alphas:
        m = float(data['BPR_baseline'][a]['APLT@10']['mean'])
        s = float(data['BPR_baseline'][a]['APLT@10']['std'])
        bl_means.append(m)
        bl_stds.append(s)
    result['BPR_baseline'] = {
        'mean': np.array(bl_means),
        'std':  np.array(bl_stds),
    }

    # ── Mitigated methods at fixed λ/τ ────────────────────────
    for method, fixed_lam in FIXED_LAMBDA.items():
        if method not in data:
            print(f"    Warning: {method} not in {dataset_name} data")
            continue

        grid = tau_grid if method == 'FAIR' else lambda_grid

        # Find index of fixed parameter in grid
        try:
            lam_idx = [round(g, 4) for g in grid].index(
                round(fixed_lam, 4)
            )
        except ValueError:
            # Try nearest
            lam_idx = int(np.argmin(np.abs(np.array(grid) - fixed_lam)))
            print(f"    {method}: λ={fixed_lam} not exact — "
                  f"using nearest {grid[lam_idx]:.2f}")

        means, stds = [], []
        for a in alphas:
            aplt_arr = data[method][a]['APLT@10']['mean']
            std_arr  = data[method][a]['APLT@10'].get('std', [0]*len(aplt_arr))

            if isinstance(aplt_arr, list):
                m = float(aplt_arr[lam_idx])
                s = float(std_arr[lam_idx]) if isinstance(std_arr, list) else 0.0
            else:
                m = float(aplt_arr)
                s = 0.0

            means.append(m)
            stds.append(s)

        result[method] = {
            'mean': np.array(means),
            'std':  np.array(stds),
        }

    return result, [float(a) for a in alphas]


def load_fallback(dataset_name):
    """Verified values from analysis — used if JSON files not found."""
    fallbacks = {
        'MovieLens': {
            'BPR_baseline': [0.1914, 0.1328, 0.0849, 0.0596, 0.0425],
            'BinaryxQuAD':  [0.2236, 0.1737, 0.1337, 0.1102, 0.0923],
            'SmoothxQuAD':  [0.4485, 0.4102, 0.3476, 0.2906, 0.2420],
            'FAIR':         [0.6045, 0.5398, 0.4571, 0.3897, 0.2901],
        },
        'AmazonBooks': {
            'BPR_baseline': [0.1027, 0.0759, 0.0480, 0.0346, 0.0248],
            'BinaryxQuAD':  [0.1794, 0.1273, 0.1060, 0.0927, 0.0885],
            'SmoothxQuAD':  [0.4636, 0.4294, 0.3565, 0.3011, 0.2451],
            'FAIR':         [0.6279, 0.5603, 0.4572, 0.3824, 0.2977],
        },
    }
    fb = fallbacks[dataset_name]
    return {k: {'mean': np.array(v), 'std': np.zeros(5)}
            for k, v in fb.items()}


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────
def plot_figure2(data_ml, alphas_ml, data_ab, alphas_ab, save_path):
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(9.5, 3.8),
        gridspec_kw={'wspace': 0.35}
    )

    panels = [
        (ax_l, data_ml, alphas_ml, 'MovieLens-1M',    '(a)'),
        (ax_r, data_ab, alphas_ab, 'Amazon Books',     '(b)'),
    ]

    for ax, data, alphas, dataset_name, panel_label in panels:
        alpha_arr = np.array(alphas)

        for method_key, style in METHODS.items():
            if method_key not in data:
                continue
            means = data[method_key]['mean']
            stds  = data[method_key]['std']

            ax.plot(alpha_arr, means,
                    color=style['color'],
                    linewidth=style['lw'],
                    linestyle=style['ls'],
                    marker=style['marker'],
                    markersize=6.5,
                    markerfacecolor='white',
                    markeredgewidth=1.8,
                    markeredgecolor=style['color'],
                    label=style['label'],
                    zorder=style['zorder'])

            # Shaded ±1 std band (usually narrow but shows stability)
            if stds.max() > 0.001:
                ax.fill_between(alpha_arr,
                                means - stds, means + stds,
                                alpha=0.12, color=style['color'],
                                zorder=style['zorder'] - 1)

        # Axis formatting
        ax.set_xlabel('Bias level α', labelpad=6)
        ax.set_ylabel('APLT@10', labelpad=6)
        ax.set_title(f'{panel_label} {dataset_name}',
                     pad=8, fontweight='bold')
        ax.set_xticks(alpha_arr)
        ax.set_xticklabels([str(a) for a in alphas])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:.2f}')
        )
        ax.grid(True, linestyle='--', linewidth=0.6,
                alpha=0.4, color='gray')

        # Dynamic y-axis — some headroom above FA*IR ceiling
        all_means = np.concatenate([v['mean'] for v in data.values()])
        y_max = all_means.max()
        y_min = all_means.min()
        ax.set_ylim(max(0, y_min - 0.02),
                    y_max + (y_max - y_min) * 0.18)

        # ── Decline annotations: show % drop for FA*IR and Baseline ──
        for method_key in ['FAIR', 'BPR_baseline']:
            if method_key not in data:
                continue
            means = data[method_key]['mean']
            pct_drop = (means[-1] - means[0]) / means[0] * 100
            # Annotate above the last point
            ax.annotate(
                f'{pct_drop:.0f}%',
                xy=(alpha_arr[-1], means[-1]),
                xytext=(6, 0), textcoords='offset points',
                fontsize=8, color=METHODS[method_key]['color'],
                va='center', fontweight='bold'
            )

    # Single legend below both panels
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=4,
               bbox_to_anchor=(0.5, -0.13),
               framealpha=0.9,
               edgecolor='#cccccc',
               fontsize=9.5)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Figure 2 — Structural Ceiling (BPR, both datasets)")
    print("=" * 58)

    results = {}
    alphas  = {}

    for dataset, json_path in STAGE2_RESULTS.items():
        print(f"\nLoading {dataset} ...")
        if os.path.exists(json_path):
            data, alp = load_aplt(json_path, dataset)
            results[dataset] = data
            alphas[dataset]  = alp
            for method, vals in data.items():
                print(f"  {method}: APLT α=1.0={vals['mean'][0]:.3f} "
                      f"→ α=2.0={vals['mean'][-1]:.3f}")
        else:
            print(f"  JSON not found — using fallback values")
            results[dataset] = load_fallback(dataset)
            alphas[dataset]  = ALPHA_LEVELS

    pdf_path = os.path.join(OUTPUT_DIR, 'fig2_structural_ceiling_bpr.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig2_structural_ceiling_bpr.png')

    print("\nGenerating figure ...")
    plot_figure2(
        results['MovieLens'],   alphas['MovieLens'],
        results['AmazonBooks'], alphas['AmazonBooks'],
        pdf_path
    )
    plot_figure2(
        results['MovieLens'],   alphas['MovieLens'],
        results['AmazonBooks'], alphas['AmazonBooks'],
        png_path
    )

    print("\nDone.")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()