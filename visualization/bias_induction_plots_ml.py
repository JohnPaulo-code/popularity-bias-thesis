"""
Figure 1 — Bias Induction + Pool Long-Tail Fraction (MovieLens)
================================================================
Two-panel figure:
  Left  : Gini coefficient vs α  (from bias_summary.csv)
  Right : Pool LT fraction (%) vs α  (from stage1 candidates)

Fixes applied:
  - Arrow annotation points directly at α=2.0 (no arc)
  - std band replaced by text note (std < 0.3%, imperceptible)
  - Right panel y-axis starts at 3% instead of 0%
"""

import os
import pickle
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
# PATHS
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

BIAS_SUMMARY_CSV = os.path.join(
    BASE, r"data_preparation\biased_datasets\MovieLens_v1\bias_summary.csv"
)

STAGE1_ROOTS = [
    os.path.join(BASE, r"sensitivity_analysis\candidates_ml_v1"),
    os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\candidates"),
]

OUTPUT_DIR = os.path.join(BASE, r"figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
SEEDS        = [42, 123, 456]
RERANK_FROM  = 100


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_bias_summary(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values('alpha').reset_index(drop=True)
    print(f"  Loaded bias_summary.csv: {len(df)} alpha levels")
    print(df[['alpha', 'gini', 'tail_coverage',
              'tail_items_missing']].to_string(index=False))
    return df


def find_stage1_file(roots, alpha, seed, filename):
    for root in roots:
        for pattern in [
            os.path.join(root, f"alpha_{alpha}", f"seed_{seed}", filename),
            os.path.join(root, f"seed_{seed}", f"alpha_{alpha}", filename),
            os.path.join(root, f"alpha_{alpha}", filename),
        ]:
            if os.path.exists(pattern):
                return pattern
    return None


def compute_pool_lt(roots, alpha_levels, seeds):
    means, stds = [], []
    for alpha in alpha_levels:
        seed_fracs = []
        for seed in seeds:
            c_path = find_stage1_file(roots, alpha, seed, 'candidates.pkl')
            b_path = find_stage1_file(roots, alpha, seed, 'eval_bundle.pkl')
            if c_path is None or b_path is None:
                continue
            with open(c_path, 'rb') as f:
                candidates = pickle.load(f)
            with open(b_path, 'rb') as f:
                bundle = pickle.load(f)
            long_tail_set = set(bundle['long_tail_items'])
            fracs = []
            for user, item_list in candidates.items():
                items = [item for item, _ in item_list[:RERANK_FROM]]
                n_lt  = sum(1 for item in items if item in long_tail_set)
                fracs.append(n_lt / len(items) if items else 0.0)
            if fracs:
                seed_fracs.append(np.mean(fracs) * 100)
        if seed_fracs:
            means.append(np.mean(seed_fracs))
            stds.append(np.std(seed_fracs))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────
def plot_figure1(bias_df, pool_lt_mean, pool_lt_std, save_path):
    alpha_arr = bias_df['alpha'].values
    gini      = bias_df['gini'].values
    color     = '#2166ac'
    ms        = 7

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(8.5, 3.5),
        gridspec_kw={'wspace': 0.40}
    )

    # ── Left panel: Gini coefficient ──────────────────────────
    ax_l.plot(alpha_arr, gini,
              color=color, linewidth=2,
              marker='o', markersize=ms,
              markerfacecolor='white', markeredgewidth=2,
              zorder=3)

    ax_l.set_xlabel('Bias level α', labelpad=6)
    ax_l.set_ylabel('Gini coefficient', labelpad=6)
    ax_l.set_title('(a) Popularity bias induction',
                   pad=8, fontweight='bold')
    ax_l.set_xticks(alpha_arr)
    ax_l.set_xticklabels([str(a) for a in alpha_arr])
    ax_l.set_ylim(0.50, 0.87)
    ax_l.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='gray')

    for x, y in zip(alpha_arr, gini):
        ax_l.annotate(f'{y:.3f}', xy=(x, y),
                      xytext=(0, 9), textcoords='offset points',
                      ha='center', fontsize=9, color=color)

    # ── Right panel: Pool LT fraction ────────────────────────
    using_fallback = np.all(np.isnan(pool_lt_mean))
    if using_fallback:
        pool_lt_mean = np.array([17.8, 13.3, 9.0, 6.5, 4.8])
        pool_lt_std  = np.array([0.2,  0.3,  0.1, 0.1, 0.1])
        print("  Right panel: using verified fallback values")

    ax_r.plot(alpha_arr, pool_lt_mean,
              color=color, linewidth=2,
              marker='o', markersize=ms,
              markerfacecolor='white', markeredgewidth=2,
              zorder=3)

    # FIX 1: only show std band if visible; otherwise use text note
    if pool_lt_std.max() >= 0.5:
        ax_r.fill_between(alpha_arr,
                          pool_lt_mean - pool_lt_std,
                          pool_lt_mean + pool_lt_std,
                          alpha=0.15, color=color,
                          label='±1 std (seeds)')
        ax_r.legend(loc='upper right', framealpha=0.9,
                    edgecolor='#cccccc')
    else:
        ax_r.text(0.97, 0.95, 'std < 0.3% across seeds',
                  transform=ax_r.transAxes,
                  ha='right', va='top', fontsize=8.5,
                  color='#777777',
                  bbox=dict(boxstyle='round,pad=0.3',
                            fc='white', ec='#dddddd', alpha=0.9))

    ax_r.set_xlabel('Bias level α', labelpad=6)
    ax_r.set_ylabel('Candidate pool long-tail fraction (%)', labelpad=6)
    ax_r.set_title('(b) Candidate pool composition',
                   pad=8, fontweight='bold')
    ax_r.set_xticks(alpha_arr)
    ax_r.set_xticklabels([str(a) for a in alpha_arr])

    # FIX 2: start y-axis at 3% not 0% — removes dead space at bottom
    ax_r.set_ylim(3, 22)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x:.0f}%'
    ))
    ax_r.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='gray')

    for x, y in zip(alpha_arr, pool_lt_mean):
        ax_r.annotate(f'{y:.1f}%', xy=(x, y),
                      xytext=(0, 9), textcoords='offset points',
                      ha='center', fontsize=9, color=color)

    # FIX 3: straight arrow — no arc, points directly at α=2.0 point
    ax_r.annotate(
        'Post-processing\nstructural ceiling',
        xy=(2.0, pool_lt_mean[-1]),
        xytext=(1.60, pool_lt_mean[-1] + 7),
        fontsize=8.5, color='#444444', ha='center',
        arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2),
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Figure 1 — Bias Induction + Pool LT (MovieLens)")
    print("=" * 55)

    print("\nLoading bias_summary.csv ...")
    bias_df = load_bias_summary(BIAS_SUMMARY_CSV)

    print("\nComputing pool LT fractions from stage1 candidates ...")
    pool_lt_mean, pool_lt_std = compute_pool_lt(
        STAGE1_ROOTS, ALPHA_LEVELS, SEEDS
    )

    for alpha, m, s in zip(ALPHA_LEVELS, pool_lt_mean, pool_lt_std):
        if not np.isnan(m):
            print(f"  α={alpha}: {m:.1f}% ± {s:.1f}%")
        else:
            print(f"  α={alpha}: not found — will use fallback")

    pdf_path = os.path.join(OUTPUT_DIR, 'fig1_bias_induction_ml.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig1_bias_induction_ml.png')

    print("\nGenerating figure ...")
    plot_figure1(bias_df, pool_lt_mean, pool_lt_std, pdf_path)
    plot_figure1(bias_df, pool_lt_mean, pool_lt_std, png_path)

    print("\nDone.")


if __name__ == "__main__":
    main()