"""
Cross-Model Pool Position Analysis — 2 × 3 Subplot Figure
===========================================================
Shows where long-tail items sit in the candidate pool
for each model architecture BEFORE mitigation.

Layout:
  Row 1 : MovieLens-1M   — BPR | LightGCN | NeuMF
  Row 2 : Amazon Books   — BPR | LightGCN | NeuMF

Y-axis: share of pool LT items at each position (%)
        = per_pos / total_pool_LT × 100
        All six subplots share the same y-axis scale,
        enabling direct model comparison.

A flat curve means LT items are spread evenly across the pool.
A rising curve means LT items are pushed toward Z-3 (deep pool),
making post-processing promotion more costly.
"""

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   10.5,
    'axes.labelsize':   10,
    'xtick.labelsize':  8.5,
    'ytick.labelsize':  8.5,
    'legend.fontsize':  9.5,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

BPR_ROOTS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\candidates_ml_v1"),
    'AmazonBooks': os.path.join(BASE, r"sensitivity_analysis\candidates_ab_v1"),
}
CROSS_ROOT = os.path.join(BASE, r"cross_model")

DATASETS     = ['MovieLens', 'AmazonBooks']
MODELS       = ['BPR', 'LightGCN', 'NeuMF']
MODEL_LABELS = {'BPR': 'BPR', 'LightGCN': 'LightGCN', 'NeuMF': 'NeuMF'}
DS_LABELS    = {'MovieLens': 'MovieLens-1M', 'AmazonBooks': 'Amazon Books'}

OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
SEEDS        = [42, 123, 456]
RERANK_FROM  = 100

ZONES = [
    ('Z-1',  1,  10,  '#b7e4c7'),
    ('Z-2', 11,  50,  '#ffe08a'),
    ('Z-3', 51, 100,  '#f4a0a0'),
]

ALPHA_COLORS = {
    1.0: '#d73027', 1.2: '#fc8d59', 1.5: '#4dac26',
    1.75: '#4575b4', 2.0: '#762a83',
}
ALPHA_LABELS = {
    1.0:'α=1.0', 1.2:'α=1.2', 1.5:'α=1.5',
    1.75:'α=1.75', 2.0:'α=2.0',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# FILE FINDER
# ─────────────────────────────────────────────────────────────
def find_file(root, alpha, seed, filename):
    patterns = [
        os.path.join(root, f"alpha_{alpha}", f"seed_{seed}", filename),
        os.path.join(root, f"seed_{seed}", f"alpha_{alpha}", filename),
        os.path.join(root, f"alpha_{alpha}", filename),
        os.path.join(root, f"alpha_{float(alpha)}", f"seed_{seed}", filename),
        os.path.join(root,
                     f"alpha_{int(alpha) if float(alpha).is_integer() else alpha}",
                     f"seed_{seed}", filename),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    return None


def get_root(model, dataset):
    if model == 'BPR':
        return BPR_ROOTS[dataset]
    ds_folder = 'MovieLens' if dataset == 'MovieLens' else 'AmazonBooks'
    return os.path.join(CROSS_ROOT, model, ds_folder)


# ─────────────────────────────────────────────────────────────
# COMPUTE PER-POSITION STATS
# ─────────────────────────────────────────────────────────────
def compute_per_pos(root, alpha_levels, seeds):
    """Returns {alpha: array[100]} — mean LT items per position per user."""
    results = {}
    for alpha in alpha_levels:
        seed_arrays = []
        for seed in seeds:
            c_path = find_file(root, alpha, seed, 'candidates.pkl')
            b_path = find_file(root, alpha, seed, 'eval_bundle.pkl')
            if c_path is None or b_path is None:
                continue
            with open(c_path, 'rb') as f:
                candidates = pickle.load(f)
            with open(b_path, 'rb') as f:
                bundle = pickle.load(f)
            lt_set  = set(bundle['long_tail_items'])
            per_pos = np.zeros(RERANK_FROM)
            n_users = 0
            for user, item_list in candidates.items():
                items = [item for item, _ in item_list[:RERANK_FROM]]
                n_users += 1
                for rank_0, item in enumerate(items):
                    if item in lt_set:
                        per_pos[rank_0] += 1
            if n_users > 0:
                seed_arrays.append(per_pos / n_users)
        if seed_arrays:
            results[alpha] = np.mean(seed_arrays, axis=0)
    return results


# ─────────────────────────────────────────────────────────────
# COMPUTE GLOBAL Y-MAX ACROSS ALL SUBPLOTS
# ─────────────────────────────────────────────────────────────
def compute_global_ymax(all_data, smooth_fn):
    """
    Normalise every curve as % of pool LT items and find the
    maximum across all subplots — used to set a shared y-axis.
    """
    global_ymax = 0.0
    for per_pos_data in all_data.values():
        for alpha, arr in per_pos_data.items():
            smoothed = smooth_fn(arr)
            total    = smoothed.sum()
            if total > 0:
                normed      = smoothed / total * 100
                global_ymax = max(global_ymax, normed.max())
    return global_ymax * 1.15   # 15% headroom


# ─────────────────────────────────────────────────────────────
# DRAW ONE SUBPLOT
# ─────────────────────────────────────────────────────────────
def draw_subplot(ax, per_pos_data, smooth_fn, global_ymax,
                 show_ylabel, show_xlabel):

    # Zone backgrounds
    for label, lo, hi, color in ZONES:
        ax.axvspan(lo - 0.5, hi + 0.5,
                   alpha=0.28, color=color, zorder=0)

    # Zone boundary lines
    for boundary in [10.5, 50.5]:
        ax.axvline(boundary, color='#555555', linewidth=1.4,
                   linestyle='--', alpha=0.80, zorder=2)

    # Density curves — normalised as % of pool LT items
    has_data  = False
    positions = np.arange(1, RERANK_FROM + 1)

    for alpha in ALPHA_LEVELS:
        if alpha not in per_pos_data:
            continue
        smoothed = smooth_fn(per_pos_data[alpha])
        total    = smoothed.sum()
        if total > 0:
            smoothed = smoothed / total * 100   # ← normalise to %
        ax.plot(positions, smoothed,
                color=ALPHA_COLORS[alpha], linewidth=1.8, zorder=3)
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'Data not found',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='gray')
        return

    # Shared y-axis
    ax.set_ylim(0, global_ymax)

    # Z labels inside plot — anchored near the x-axis
    zone_centres = [5.5, 30.5, 75.5]
    y_label_pos  = global_ymax * 0.04
    for (zlabel, _, _, _), xc in zip(ZONES, zone_centres):
        ax.text(xc, y_label_pos, zlabel,
                ha='center', va='bottom',
                fontsize=8.5, fontweight='bold',
                color='#333333', zorder=5)

    ax.set_xlim(0.5, 100.5)
    ax.grid(True, linestyle='--', linewidth=0.4,
            alpha=0.35, color='gray', zorder=1)

    if show_xlabel:
        ax.set_xlabel('Position in candidate pool', labelpad=5)
    if show_ylabel:
        ax.set_ylabel('Share of pool LT items (%)', labelpad=5)

    ax.tick_params(axis='both', labelsize=8.5)


# ─────────────────────────────────────────────────────────────
# FULL 2 × 3 FIGURE
# ─────────────────────────────────────────────────────────────
def plot_crossmodel_figure(all_data, save_path):
    try:
        from scipy.ndimage import gaussian_filter1d
        smooth_fn = lambda x: gaussian_filter1d(x, sigma=2.5)
    except ImportError:
        smooth_fn = lambda x: x

    # Compute shared y ceiling across all 6 subplots
    global_ymax = compute_global_ymax(all_data, smooth_fn)
    print(f"  Global y-max (normalised): {global_ymax:.2f}%")

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 7),
        gridspec_kw={'hspace': 0.38, 'wspace': 0.30},
    )
    fig.subplots_adjust(bottom=0.14, left=0.09, right=0.98, top=0.92)

    for row_i, dataset in enumerate(DATASETS):
        for col_i, model in enumerate(MODELS):
            ax           = axes[row_i, col_i]
            per_pos_data = all_data.get((model, dataset), {})

            draw_subplot(
                ax, per_pos_data, smooth_fn, global_ymax,
                show_ylabel = (col_i == 0),
                show_xlabel = (row_i == 1),
            )

            # Column header — top row only
            if row_i == 0:
                ax.set_title(MODEL_LABELS[model],
                             fontsize=11, fontweight='bold', pad=6)

        # Dataset row label — rotated, far left
        fig.text(
            0.01,
            0.73 - row_i * 0.47,
            DS_LABELS[dataset],
            va='center', ha='left',
            fontsize=10.5, fontweight='bold',
            rotation=90, color='#222222',
        )

    # Horizontal separator between rows
    fig.add_artist(
        matplotlib.lines.Line2D(
            [0.07, 0.99], [0.52, 0.52],
            transform=fig.transFigure,
            color='#bbbbbb', linewidth=0.8, linestyle='--'
        )
    )

    # Shared legend at bottom
    legend_handles = [
        mlines.Line2D([0],[0],
                      color=ALPHA_COLORS[a], linewidth=2,
                      label=ALPHA_LABELS[a])
        for a in ALPHA_LEVELS
    ]
    fig.legend(
        legend_handles,
        [ALPHA_LABELS[a] for a in ALPHA_LEVELS],
        loc='lower center', ncol=5,
        bbox_to_anchor=(0.5, 0.01),
        framealpha=0.95, edgecolor='#cccccc',
        fontsize=10, handlelength=2,
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Cross-Model Pool Position Analysis (2 × 3 figure)")
    print("  Y-axis: share of pool LT items (%) — shared scale")
    print("=" * 62)

    all_data = {}
    for dataset in DATASETS:
        for model in MODELS:
            root = get_root(model, dataset)
            print(f"\n  Loading {model} / {dataset} ...")
            per_pos = compute_per_pos(root, ALPHA_LEVELS, SEEDS)
            all_data[(model, dataset)] = per_pos
            if per_pos:
                total_lt = {a: v.sum() for a, v in per_pos.items()}
                print(f"    α=1.0: {total_lt.get(1.0, 0):.2f} LT items  "
                      f"α=2.0: {total_lt.get(2.0, 0):.2f} LT items")
            else:
                print(f"    ⚠ No data found — check path structure")

    print("\nGenerating figure ...")
    import matplotlib.lines
    plot_crossmodel_figure(
        all_data,
        os.path.join(OUTPUT_DIR, 'fig_pool_position_crossmodel.pdf')
    )
    plot_crossmodel_figure(
        all_data,
        os.path.join(OUTPUT_DIR, 'fig_pool_position_crossmodel.png')
    )
    print("\nDone.")


if __name__ == "__main__":
    main()