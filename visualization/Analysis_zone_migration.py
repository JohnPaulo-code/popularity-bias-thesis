"""
Analysis A — Zone Migration
============================
Uses the EXACT same reranking classes as cross_model.py:
  from reranking.xquad_binary     import BinaryxQuAD
  from reranking.xquad_smooth     import SmoothxQuAD
  from reranking.fair_reranker_v1 import FAIRReranker

And the same build_user_data / run_xquad / run_fair helpers
copied verbatim from cross_model.py — guaranteeing identical
behaviour to the published experimental results.

For each user, each LT item in the final top-10 is traced back
to its original pool position and assigned to:
  Z-1 (1-10)   : already in baseline top-10 — free gain
  Z-2 (11-50)  : promotable zone — reranker pulled it up
  Z-3 (51-100) : deep pool — forced at high NDCG cost

Data loaded from stage1 output (candidates.pkl + eval_bundle.pkl
+ user_profiles.pkl) — EXACTLY the same files that stage2 used.
No experiment files are modified.

Outputs:
  tab_zone_migration_ml.tex / _ab.tex
  fig_zone_migration.pdf / .png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─────────────────────────────────────────────────────────────
# ADD PROJECT TO PATH — same pattern as cross_model.py
# ─────────────────────────────────────────────────────────────
BASE         = r"D:\Final_thesis\thesis_project"
PROJECT_ROOT = BASE

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reranking.xquad_binary     import BinaryxQuAD
from reranking.xquad_smooth     import SmoothxQuAD
from reranking.fair_reranker_v1 import FAIRReranker

print("  Reranking modules loaded from project.")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
STAGE1_ROOTS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\candidates_ml_v1"),
    'AmazonBooks': os.path.join(BASE, r"sensitivity_analysis\candidates_ab_v1"),
}

OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
SEEDS        = [42, 123, 456]
K            = 10
RERANK_FROM  = 100

# Nash consensus — same values used in the published experiments
NASH_PARAMS = {
    'MovieLens':   {'BinaryxQuAD': 0.4, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
    'AmazonBooks': {'BinaryxQuAD': 0.3, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
}

ZONE_BOUNDS = [(1, 10), (11, 50), (51, 100)]
ZONE_NAMES  = ['Z-1 (1–10)', 'Z-2 (11–50)', 'Z-3 (51–100)']
METHODS     = ['Baseline', 'BinaryxQuAD', 'SmoothxQuAD', 'FAIR']

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 10.5, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# FILE FINDER
# ─────────────────────────────────────────────────────────────
def find_file(root, alpha, seed, filename):
    for p in [
        os.path.join(root, f"alpha_{alpha}", f"seed_{seed}", filename),
        os.path.join(root, f"seed_{seed}", f"alpha_{alpha}", filename),
        os.path.join(root, f"alpha_{alpha}", filename),
    ]:
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────
# HELPERS — copied verbatim from cross_model.py
# ─────────────────────────────────────────────────────────────
def build_user_data(candidates, user_profiles):
    """Copied verbatim from cross_model.py — identical preprocessing."""
    user_data = {}
    for user, item_list in candidates.items():
        items  = [item for item, _ in item_list[:RERANK_FROM]]
        scores = {item: sc for item, sc in item_list[:RERANK_FROM]}
        max_s  = max(scores.values()) if scores else 1.0
        norm   = ({k: v / max_s for k, v in scores.items()}
                  if max_s > 0 else scores)
        user_data[user] = {
            'items':       items,
            'scores':      scores,
            'norm_scores': norm,
            'profile':     user_profiles.get(user, set()),
        }
    return user_data


def run_xquad(reranker_cls, user_data, long_tail, short_head, lam):
    """Copied verbatim from cross_model.py."""
    reranker = reranker_cls(
        lambda_param=lam, rerank_from=RERANK_FROM, rerank_to=K
    )
    recs = {}
    for user, ud in user_data.items():
        recs[user] = reranker.rerank(
            user_idx         = user,
            candidate_items  = ud['items'],
            base_scores      = ud['norm_scores'],
            user_profile     = ud['profile'],
            long_tail_items  = long_tail,
            short_head_items = short_head,
        )
    return recs


def run_fair(user_data, long_tail, tau):
    """Copied verbatim from cross_model.py."""
    reranker = FAIRReranker(p=tau, alpha=0.05, k=K)
    recs     = {}
    for user, ud in user_data.items():
        pm       = {item: (item in long_tail) for item in ud['items']}
        reranked = reranker.rerank(ud['items'], ud['scores'], pm)
        recs[user] = [(item, ud['scores'].get(item, 0.0))
                      for item in reranked]
    return recs


# ─────────────────────────────────────────────────────────────
# ZONE ASSIGNMENT
# ─────────────────────────────────────────────────────────────
def get_zone(pool_position_1indexed):
    for z_i, (lo, hi) in enumerate(ZONE_BOUNDS):
        if lo <= pool_position_1indexed <= hi:
            return z_i
    return len(ZONE_BOUNDS) - 1


# ─────────────────────────────────────────────────────────────
# COMPUTE ZONE MIGRATION
# ─────────────────────────────────────────────────────────────
def compute_zone_migration(root, dataset, alpha_levels, seeds):
    """
    For each (alpha, method), compute mean LT items per user
    in final top-10 broken down by origin zone in the pool.
    Returns {alpha: {method: array[4]}}  (Z1, Z2, Z3, total)
    """
    params  = NASH_PARAMS[dataset]
    results = {a: {m: np.zeros(4) for m in METHODS}
               for a in alpha_levels}

    for alpha in alpha_levels:
        print(f"    α={alpha} ...", end=' ', flush=True)
        seed_data = {m: [] for m in METHODS}

        for seed in seeds:
            c_path = find_file(root, alpha, seed, 'candidates.pkl')
            b_path = find_file(root, alpha, seed, 'eval_bundle.pkl')
            p_path = find_file(root, alpha, seed, 'user_profiles.pkl')

            if c_path is None or b_path is None:
                continue

            with open(c_path, 'rb') as f:
                candidates = pickle.load(f)
            with open(b_path, 'rb') as f:
                bundle = pickle.load(f)

            # Load user profiles — fall back to empty if not found
            if p_path:
                with open(p_path, 'rb') as f:
                    user_profiles = pickle.load(f)
            else:
                user_profiles = {}

            lt_set    = set(bundle['long_tail_items'])
            head_set  = set(bundle.get('short_head_items', []))
            user_data = build_user_data(candidates, user_profiles)

            # Build position map: user → {item: pool_position (1-indexed)}
            pos_maps = {}
            for user, item_list in candidates.items():
                pos_maps[user] = {
                    item: rank + 1
                    for rank, (item, _) in enumerate(item_list[:RERANK_FROM])
                }

            # --- Baseline ---
            baseline_recs = {
                u: ud['items'][:K] for u, ud in user_data.items()
            }

            # --- BinaryxQuAD ---
            bxq_recs = run_xquad(BinaryxQuAD, user_data,
                                  lt_set, head_set,
                                  params['BinaryxQuAD'])

            # --- SmoothxQuAD ---
            sxq_recs = run_xquad(SmoothxQuAD, user_data,
                                  lt_set, head_set,
                                  params['SmoothxQuAD'])

            # --- FA*IR ---
            fair_recs = run_fair(user_data, lt_set, params['FAIR'])

            all_recs = {
                'Baseline':    {u: r for u, r in baseline_recs.items()},
                'BinaryxQuAD': {u: [it for it, _ in r]
                                for u, r in bxq_recs.items()},
                'SmoothxQuAD': {u: [it for it, _ in r]
                                for u, r in sxq_recs.items()},
                'FAIR':        {u: [it for it, _ in r]
                                for u, r in fair_recs.items()},
            }

            # Accumulate zone counts per user
            seed_user = {m: np.zeros(4) for m in METHODS}
            n_users   = 0

            for user in user_data:
                pos_map = pos_maps.get(user, {})
                n_users += 1
                for method in METHODS:
                    top10      = all_recs[method].get(user, [])
                    zone_counts = np.zeros(4)
                    for item in top10:
                        if item in lt_set:
                            pool_pos = pos_map.get(item, RERANK_FROM)
                            z = get_zone(pool_pos)
                            zone_counts[z] += 1
                            zone_counts[3] += 1   # total
                    seed_user[method] += zone_counts

            if n_users > 0:
                for m in METHODS:
                    seed_data[m].append(seed_user[m] / n_users)

        for m in METHODS:
            if seed_data[m]:
                results[alpha][m] = np.mean(seed_data[m], axis=0)

        print("done")

    return results


# ─────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────
def print_table(dataset_name, results):
    print(f"\n{'='*76}")
    print(f"  {dataset_name} — Zone Migration")
    print(f"  (mean LT items per user in final top-10 by origin zone)")
    print(f"{'='*76}")
    print(f"{'α':>5}  {'Method':>14}  {'Z-1 (1-10)':>11}  "
          f"{'Z-2 (11-50)':>12}  {'Z-3 (51-100)':>13}  {'Total':>7}")
    print('-' * 76)
    for alpha in ALPHA_LEVELS:
        first = True
        for method in METHODS:
            r      = results[alpha][method]
            prefix = f"{alpha:>5.2f}" if first else f"{'':>5}"
            first  = False
            print(f"{prefix}  {method:>14}  {r[0]:>11.3f}  "
                  f"{r[1]:>12.3f}  {r[2]:>13.3f}  {r[3]:>7.3f}")
        print()


# ─────────────────────────────────────────────────────────────
# LATEX TABLE
# ─────────────────────────────────────────────────────────────
def write_latex(dataset_name, results, outpath):
    ds    = dataset_name.replace("AmazonBooks", "Amazon Books")
    label = "ml" if "MovieLens" in dataset_name else "ab"
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{" + ds +
        r": mean long-tail items per user in the final top-10, "
        r"broken down by their origin zone in the 100-candidate pool. "
        r"Z-1 (1--10): items already in the unmitigated top-10. "
        r"Z-2 (11--50): items promoted from the reachable zone. "
        r"Z-3 (51--100): items pulled from the deep pool. "
        r"Reranking applied using the same implementation as the "
        r"published cross-model experiments.}"
    )
    lines.append(r"\label{tab:zone_migration_" + label + r"}")
    lines.append(r"\begin{tabular}{clcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$\alpha$ & Method & "
        r"\makecell{Z-1\\(1--10)} & "
        r"\makecell{Z-2\\(11--50)} & "
        r"\makecell{Z-3\\(51--100)} & Total \\"
    )
    lines.append(r"\midrule")
    for alpha in ALPHA_LEVELS:
        lines.append(r"\multirow{4}{*}{" + f"{alpha:.2f}" + r"}")
        for i, method in enumerate(METHODS):
            r     = results[alpha][method]
            shade = r" \rowcolor{gray!6}" if i % 2 == 1 else ""
            mname = {'BinaryxQuAD': 'Binary xQuAD',
                     'SmoothxQuAD': 'Smooth xQuAD',
                     'FAIR': 'FA*IR'}.get(method, method)
            lines.append(
                f"{shade} & {mname} & "
                f"{r[0]:.3f} & {r[1]:.3f} & "
                f"{r[2]:.3f} & \\textbf{{{r[3]:.3f}}} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


# ─────────────────────────────────────────────────────────────
# FIGURE — stacked bar
# ─────────────────────────────────────────────────────────────
def plot_zone_migration(all_results, save_path):
    datasets    = ['MovieLens', 'AmazonBooks']
    ds_labels   = {'MovieLens': 'MovieLens-1M', 'AmazonBooks': 'Amazon Books'}
    m_labels    = ['Baseline', 'Binary\nxQuAD', 'Smooth\nxQuAD', 'FA*IR']
    zone_colors = ['#52b788', '#f4a261', '#e63946']
    zone_labels = ['Z-1 (1–10)', 'Z-2 (11–50)', 'Z-3 (51–100)']

    fig, axes = plt.subplots(
        2, 5, figsize=(14, 6),
        gridspec_kw={'hspace': 0.45, 'wspace': 0.22}
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.90)

    for row_i, dataset in enumerate(datasets):
        results = all_results[dataset]
        for col_i, alpha in enumerate(ALPHA_LEVELS):
            ax = axes[row_i, col_i]
            for m_i, method in enumerate(METHODS):
                r   = results[alpha][method]
                bot = 0.0
                for z in range(3):
                    ax.bar(m_i, r[z], bottom=bot,
                           color=zone_colors[z], width=0.65,
                           alpha=0.88, zorder=2)
                    bot += r[z]
            ax.set_xticks(range(len(METHODS)))
            ax.set_xticklabels(m_labels, fontsize=7.5)
            ax.set_xlim(-0.5, len(METHODS) - 0.5)
            ax.grid(True, axis='y', linestyle='--',
                    linewidth=0.4, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            if col_i == 0:
                ax.set_ylabel(
                    f'{ds_labels[dataset]}\n\nLT items in top-10',
                    labelpad=5, fontsize=9
                )
            if row_i == 0:
                ax.set_title(f'α={alpha}', fontweight='bold', pad=5)

    legend_handles = [
        Patch(color=zone_colors[z], alpha=0.88, label=zone_labels[z])
        for z in range(3)
    ]
    fig.legend(legend_handles, zone_labels,
               loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02),
               framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Analysis A — Zone Migration")
    print("  Using project reranking implementations")
    print("=" * 60)

    all_results = {}
    for dataset, root in STAGE1_ROOTS.items():
        print(f"\n{dataset}")
        results = compute_zone_migration(root, dataset,
                                         ALPHA_LEVELS, SEEDS)
        all_results[dataset] = results
        print_table(dataset, results)
        label = "ml" if "MovieLens" in dataset else "ab"
        write_latex(dataset, results,
                    os.path.join(OUTPUT_DIR,
                                 f"tab_zone_migration_{label}.tex"))

    plot_zone_migration(
        all_results,
        os.path.join(OUTPUT_DIR, 'fig_zone_migration.pdf')
    )
    plot_zone_migration(
        all_results,
        os.path.join(OUTPUT_DIR, 'fig_zone_migration.png')
    )
    print("\nDone.")


if __name__ == "__main__":
    main()