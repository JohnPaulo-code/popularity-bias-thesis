"""
Pool Position Analysis — Where Do Long-Tail Items Sit in the Candidate Pool?
"""

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

BASE = r"D:\Final_thesis\thesis_project"

STAGE1_ROOTS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\candidates_ml_v1"),
    'AmazonBooks': os.path.join(BASE, r"sensitivity_analysis\candidates_ab_v1"),
}
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


def find_file(root, alpha, seed, filename):
    for p in [
        os.path.join(root, f"alpha_{alpha}", f"seed_{seed}", filename),
        os.path.join(root, f"seed_{seed}", f"alpha_{alpha}", filename),
        os.path.join(root, f"alpha_{alpha}", filename),
    ]:
        if os.path.exists(p):
            return p
    return None


def compute_stats(root, alpha_levels, seeds):
    results = {}
    for alpha in alpha_levels:
        seed_per_pos   = []
        seed_zone_data = []

        for seed in seeds:
            c_path = find_file(root, alpha, seed, 'candidates.pkl')
            b_path = find_file(root, alpha, seed, 'eval_bundle.pkl')
            if c_path is None or b_path is None:
                continue
            with open(c_path, 'rb') as f:
                candidates = pickle.load(f)
            with open(b_path, 'rb') as f:
                bundle = pickle.load(f)
            lt_set = set(bundle['long_tail_items'])

            per_pos    = np.zeros(RERANK_FROM)
            zone_ranks = [[] for _ in ZONES]
            n_users    = 0

            for user, item_list in candidates.items():
                items = [item for item, _ in item_list[:RERANK_FROM]]
                n_users += 1
                for rank_0, item in enumerate(items):
                    rank_1 = rank_0 + 1
                    if item in lt_set:
                        per_pos[rank_0] += 1
                        for z_i, (_, lo, hi, _) in enumerate(ZONES):
                            if lo <= rank_1 <= hi:
                                zone_ranks[z_i].append(rank_1)

            if n_users > 0:
                seed_per_pos.append(per_pos / n_users)
                seed_zone_data.append(
                    [(np.mean(zone_ranks[z]) if zone_ranks[z] else 0.0)
                     for z in range(len(ZONES))]
                )

        if seed_per_pos:
            mean_per_pos  = np.mean(seed_per_pos, axis=0)
            mean_zone_rnk = np.mean(seed_zone_data, axis=0)
            zone_stats = []
            for z_i, (_, lo, hi, _) in enumerate(ZONES):
                count     = mean_per_pos[lo-1:hi].sum()
                mean_rank = float(mean_zone_rnk[z_i])
                zone_stats.append((count, mean_rank))
            results[alpha] = {'per_pos': mean_per_pos, 'zones': zone_stats}

    return results


def print_table(dataset_name, stats):
    print(f"\n{'='*72}")
    print(f"  {dataset_name}  —  count (mean rank within zone)")
    print(f"{'='*72}")
    print(f"{'α':>5}  {'Z-1 (1-10)':>18}  "
          f"{'Z-2 (11-50)':>18}  {'Z-3 (51-100)':>19}")
    print('-'*72)
    for alpha in ALPHA_LEVELS:
        if alpha not in stats:
            continue
        z = stats[alpha]['zones']
        print(f"{alpha:>5.2f}  "
              f"{z[0][0]:5.2f} (rank {z[0][1]:4.1f})  "
              f"{z[1][0]:5.2f} (rank {z[1][1]:5.1f})  "
              f"{z[2][0]:5.2f} (rank {z[2][1]:5.1f})")


def write_latex(dataset_name, stats, outpath):
    label = "ml" if "MovieLens" in dataset_name else "ab"
    ds    = dataset_name.replace("AmazonBooks", "Amazon Books")
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{" + ds +
        r": mean long-tail items per user and their mean rank within each "
        r"zone of the 100-candidate pool (format: \emph{count} (mean rank)). "
        r"Z-1 = positions 1--10 (baseline top-10), "
        r"Z-2 = positions 11--50 (promotable), "
        r"Z-3 = positions 51--100 (deep pool).}"
    )
    lines.append(r"\label{tab:pool_position_" + label + r"}")
    lines.append(r"\begin{tabular}{cccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$\alpha$ & "
        r"\makecell{Z-1 (1--10)\\count (mean rank)} & "
        r"\makecell{Z-2 (11--50)\\count (mean rank)} & "
        r"\makecell{Z-3 (51--100)\\count (mean rank)} \\"
    )
    lines.append(r"\midrule")
    for i, alpha in enumerate(ALPHA_LEVELS):
        if alpha not in stats:
            continue
        z     = stats[alpha]['zones']
        shade = r"\rowcolor{gray!8} " if i % 2 == 0 else ""
        lines.append(
            f"{shade}{alpha:.2f} & "
            f"{z[0][0]:.2f} ({z[0][1]:.1f}) & "
            f"{z[1][0]:.2f} ({z[1][1]:.1f}) & "
            f"{z[2][0]:.2f} ({z[2][1]:.1f}) \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


def plot_figure(all_stats, save_path):
    try:
        from scipy.ndimage import gaussian_filter1d
        smooth_fn = lambda x: gaussian_filter1d(x, sigma=2.5)
    except ImportError:
        smooth_fn = lambda x: x

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={'wspace': 0.36}
    )
    fig.subplots_adjust(bottom=0.26)

    panels = [
        (ax_l, all_stats.get('MovieLens',   {}), '(a) MovieLens-1M'),
        (ax_r, all_stats.get('AmazonBooks', {}), '(b) Amazon Books'),
    ]

    for ax, stats, title in panels:

        # ── Zone backgrounds ──────────────────────────────────
        for label, lo, hi, color in ZONES:
            ax.axvspan(lo - 0.5, hi + 0.5,
                       alpha=0.30, color=color, zorder=0)

        # ── Zone boundary lines ───────────────────────────────
        for boundary in [10.5, 50.5]:
            ax.axvline(boundary, color='#555555', linewidth=1.6,
                       linestyle='--', alpha=0.85, zorder=2)

        # ── Density curves ────────────────────────────────────
        y_max     = 0
        y_alpha20 = {}    # store α=2.0 values to anchor zone labels

        for alpha in ALPHA_LEVELS:
            if alpha not in stats:
                continue
            smoothed  = smooth_fn(stats[alpha]['per_pos'])
            positions = np.arange(1, RERANK_FROM + 1)
            ax.plot(positions, smoothed,
                    color=ALPHA_COLORS[alpha], linewidth=2.2,
                    zorder=3)
            y_max = max(y_max, smoothed.max())
            if alpha == 2.0:
                y_alpha20 = dict(zip(positions, smoothed))

        ax.set_ylim(0, y_max * 1.12)

        # ── Zone labels: Z-1, Z-2, Z-3 inside plot ───────────
        # Place each label just above the x-axis, centred in its zone
        # Position at y = small fixed offset above 0
        zone_centres = [5.5, 30.5, 75.5]
        y_label_pos  = y_max * 0.04    # just above the x-axis line

        for (zlabel, lo, hi, _), xc in zip(ZONES, zone_centres):
            ax.text(xc, y_label_pos, zlabel,
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    color='#333333', zorder=5)

        # ── Axis formatting ───────────────────────────────────
        ax.set_xlabel('Position in candidate pool', labelpad=8)
        ax.set_ylabel('Mean LT items per user', labelpad=6)
        ax.set_title(title, pad=8, fontweight='bold')
        ax.set_xlim(0.5, 100.5)
        ax.grid(True, linestyle='--', linewidth=0.5,
                alpha=0.35, color='gray', zorder=1)

    # ── Shared legend at the bottom ────────────────────────────
    legend_handles = [
        mlines.Line2D([0],[0],
                      color=ALPHA_COLORS[a], linewidth=2.2,
                      label=ALPHA_LABELS[a])
        for a in ALPHA_LEVELS
    ]
    fig.legend(
        legend_handles,
        [ALPHA_LABELS[a] for a in ALPHA_LEVELS],
        loc='lower center',
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=10,
        handlelength=2,
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure → {save_path}")


def main():
    print("="*60)
    print("  Pool Position Analysis")
    print("="*60)

    all_stats = {}
    for dataset, root in STAGE1_ROOTS.items():
        print(f"\nComputing {dataset} ...")
        stats = compute_stats(root, ALPHA_LEVELS, SEEDS)
        all_stats[dataset] = stats
        print_table(dataset, stats)
        label = "ml" if "MovieLens" in dataset else "ab"
        write_latex(dataset, stats,
                    os.path.join(OUTPUT_DIR, f"tab_pool_position_{label}.tex"))

    print("\nGenerating figure ...")
    plot_figure(all_stats, os.path.join(OUTPUT_DIR, 'fig_pool_position.pdf'))
    plot_figure(all_stats, os.path.join(OUTPUT_DIR, 'fig_pool_position.png'))
    print("\nDone.")


if __name__ == "__main__":
    main()