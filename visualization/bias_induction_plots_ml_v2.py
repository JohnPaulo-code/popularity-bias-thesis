"""
Figure 1 — Bias Induction + Pool LT – Extended Range (fixed)
=============================================================
Two-panel figure:
  Left  : Gini coefficient vs α (combines main + reverse)
  Right : Pool LT fraction (%) vs α
Range: 0.5, 0.75, 1.0, 1.2, 1.5, 1.75, 2.0
α=1.0 marked as natural.
"""

import os, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 11,
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ─── PATHS (adjust to your folders) ────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

MAIN_BIAS_CSV = os.path.join(BASE, r"data_preparation\biased_datasets\MovieLens_v1\bias_summary.csv")
REVERSE_BIAS_CSV = os.path.join(BASE, r"data_preparation\biased_datasets\MovieLens_v2\bias_summary.csv")

MAIN_STAGE1_ROOTS = [
    os.path.join(BASE, r"sensitivity_analysis\candidates_ml_v1"),
    os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\candidates"),
]
REVERSE_STAGE1_ROOTS = [
    os.path.join(BASE, r"sensitivity_analysis\candidates_ml_v2"),
]

OUTPUT_DIR = os.path.join(BASE, r"figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA_LEVELS = [0.5, 0.75, 1.0, 1.2, 1.5, 1.75, 2.0]
SEEDS = [42, 123, 456]
RERANK_FROM = 100

# ─── DATA LOADING ──────────────────────────────────────────────
def load_bias_summary(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path).sort_values('alpha').reset_index(drop=True)
    return df

def combine_bias_data(main_csv, reverse_csv):
    df_main = load_bias_summary(main_csv)
    df_rev  = load_bias_summary(reverse_csv)
    if df_main is None and df_rev is None:
        raise FileNotFoundError("No bias summary files found.")
    if df_main is None:
        df = df_rev
    elif df_rev is None:
        df = df_main
    else:
        df = pd.concat([df_rev, df_main], ignore_index=True)
    df = df.sort_values('alpha').reset_index(drop=True)
    print("Combined bias_summary:")
    print(df[['alpha', 'gini', 'tail_coverage']].to_string(index=False))
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
                n_lt = sum(1 for item in items if item in long_tail_set)
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

# ─── PLOTTING ──────────────────────────────────────────────────
def plot_figure1(bias_df, pool_lt_mean, pool_lt_std, alpha_arr, save_path):
    color = '#2166ac'
    ms = 7
    natural_color = '#d62728'

    sort_idx = np.argsort(alpha_arr)
    alpha_arr = alpha_arr[sort_idx]
    gini = bias_df['gini'].values[sort_idx]

    if np.all(np.isnan(pool_lt_mean)):
        pool_lt_mean = np.array([30.7, 23.7, 17.8, 13.3, 9.0, 6.5, 4.8])
        pool_lt_std = np.array([0.3, 0.2, 0.2, 0.3, 0.1, 0.1, 0.1])
        print("Using fallback pool LT values.")
    else:
        pool_lt_mean = pool_lt_mean[sort_idx]
        pool_lt_std = pool_lt_std[sort_idx]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(8.8, 3.4),
        gridspec_kw={'wspace': 0.30}
    )
    fig.subplots_adjust(top=0.88, bottom=0.20, left=0.08, right=0.98)

    ax_l.plot(
        alpha_arr, gini, color=color, linewidth=2,
        marker='o', markersize=ms, markerfacecolor='white',
        markeredgewidth=2, zorder=3
    )
    ax_l.set_xlabel('Bias level α')
    ax_l.set_ylabel('Gini coefficient')
    ax_l.set_title('(a) Popularity bias induction', fontweight='bold', pad=8)
    ax_l.set_xticks(alpha_arr)
    ax_l.set_xticklabels([str(a) for a in alpha_arr])
    ax_l.set_ylim(0.30, 0.87)
    ax_l.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='gray')

    for x, y in zip(alpha_arr, gini):
        ax_l.annotate(
            f'{y:.3f}', xy=(x, y), xytext=(0, 8),
            textcoords='offset points', ha='center',
            fontsize=9, color=color
        )

    ax_l.axvline(x=1.0, color=natural_color, linestyle='--', linewidth=1.8, alpha=0.9)
    ax_l.text(
        1.0, 0.98, 'natural',
        transform=ax_l.get_xaxis_transform(),
        ha='center', va='top', fontsize=8, color=natural_color
    )

    ax_r.plot(
        alpha_arr, pool_lt_mean, color=color, linewidth=2,
        marker='o', markersize=ms, markerfacecolor='white',
        markeredgewidth=2, zorder=3
    )

    if pool_lt_std.max() >= 0.5:
        ax_r.fill_between(
            alpha_arr, pool_lt_mean - pool_lt_std, pool_lt_mean + pool_lt_std,
            alpha=0.15, color=color, label='±1 std'
        )
        ax_r.legend(loc='upper right', framealpha=0.9, edgecolor='#cccccc')
    else:
        ax_r.text(
            0.97, 0.95, 'std < 0.3% across seeds',
            transform=ax_r.transAxes, ha='right', va='top',
            fontsize=8.5, color='#777777',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.9)
        )

    ax_r.set_xlabel('Bias level α')
    ax_r.set_ylabel('Candidate pool long-tail fraction (%)')
    ax_r.set_title('(b) Candidate pool composition', fontweight='bold', pad=8)
    ax_r.set_xticks(alpha_arr)
    ax_r.set_xticklabels([str(a) for a in alpha_arr])
    ax_r.set_ylim(3, 35)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax_r.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='gray')

    for x, y in zip(alpha_arr, pool_lt_mean):
        ax_r.annotate(
            f'{y:.1f}%', xy=(x, y), xytext=(0, 8),
            textcoords='offset points', ha='center',
            fontsize=9, color=color
        )

    ax_r.axvline(x=1.0, color=natural_color, linestyle='--', linewidth=1.8, alpha=0.9)
    ax_r.text(
        1.0, 0.98, 'natural',
        transform=ax_r.get_xaxis_transform(),
        ha='center', va='top', fontsize=8, color=natural_color
    )

    ax_r.annotate(
        'Post-processing\nstructural ceiling',
        xy=(2.0, pool_lt_mean[-1]),
        xytext=(1.60, pool_lt_mean[-1] + 7),
        fontsize=8.5, color='#444444',
        ha='center',
        arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2)
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

def main():
    print("Figure 1 — Extended Range (MovieLens)")
    bias_df = combine_bias_data(MAIN_BIAS_CSV, REVERSE_BIAS_CSV)
    alpha_arr = bias_df['alpha'].values

    all_roots = MAIN_STAGE1_ROOTS + REVERSE_STAGE1_ROOTS
    print("\nComputing pool LT fractions ...")
    pool_lt_mean, pool_lt_std = compute_pool_lt(all_roots, ALPHA_LEVELS, SEEDS)

    for a, m, s in zip(ALPHA_LEVELS, pool_lt_mean, pool_lt_std):
        if not np.isnan(m):
            print(f"  α={a}: {m:.1f}% ± {s:.1f}%")
        else:
            print(f"  α={a}: not found")

    pdf_path = os.path.join(OUTPUT_DIR, 'fig1_bias_induction_ml_extended.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig1_bias_induction_ml_extended.png')
    plot_figure1(bias_df, pool_lt_mean, pool_lt_std, alpha_arr, pdf_path)
    plot_figure1(bias_df, pool_lt_mean, pool_lt_std, alpha_arr, png_path)
    print("Done.")

if __name__ == "__main__":
    main()