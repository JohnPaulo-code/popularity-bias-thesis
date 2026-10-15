"""
Nash Bargaining Parameter Selection
=====================================
Identifies optimal λ/τ per method using Nash bargaining product:
    Nash(p) = (NDCG(p) / NDCG_base) × (APLT(p) / APLT_base)

Two outputs:
  1. Per-alpha optimal (for sensitivity analysis discussion)
  2. Cross-alpha consensus (single parameter for cross-model experiment)

The cross-alpha consensus is the parameter that maximises the
mean Nash score averaged across all alpha levels — the value
most robust to varying bias conditions, used unchanged for
LightGCN and NeuMF.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────
# PATHS — update both datasets
# ─────────────────────────────────────────────────────────────
DATASETS = {
    'MovieLens': {
        'aggregated': r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json",
        'output':     r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml_v1\nash_selection",
    },
    'AmazonBooks': {
        'aggregated': r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ab_v1\enhanced_analysis\aggregated_results.json",
        'output':     r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ab_v1\nash_selection",
    },
}

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — 5 alpha levels
# ─────────────────────────────────────────────────────────────
ALPHA_LEVELS  = ['1.0', '1.2', '1.5', '1.75', '2.0']

ALPHA_COLORS  = {
    '1.0':  '#c6dbef',
    '1.2':  '#6baed6',
    '1.5':  '#2171b5',
    '1.75': '#08519c',
    '2.0':  '#08306b',
}
ALPHA_LABELS  = {a: f'α={a}' for a in ALPHA_LEVELS}

METHODS = {
    'BinaryxQuAD': {'label': 'Binary xQuAD', 'param': 'λ', 'color': '#1f77b4', 'marker': 'o'},
    'SmoothxQuAD': {'label': 'Smooth xQuAD', 'param': 'λ', 'color': '#ff7f0e', 'marker': 's'},
    'FAIR':         {'label': 'FA*IR',         'param': 'τ', 'color': '#2ca02c', 'marker': '^'},
}


# ─────────────────────────────────────────────────────────────
# NASH CORE
# ─────────────────────────────────────────────────────────────
def nash_score(ndcg, aplt, base_ndcg, base_aplt):
    """
    Nash bargaining product using relative improvement.
    = (NDCG / NDCG_baseline) × (APLT / APLT_baseline)

    λ=0 → Nash = 1.0 (pure baseline)
    λ>0 → Nash > 1.0 if both metrics improve
           or Nash > 1.0 if APLT gain outweighs NDCG loss

    Maximising this selects the parameter that best balances
    accuracy preservation with fairness improvement.
    """
    return (ndcg / base_ndcg) * (aplt / base_aplt)


def compute_nash(data: dict) -> dict:
    """Compute per-alpha and cross-alpha Nash results for all methods."""
    lambda_grid = data['lambda_grid']
    tau_grid    = data.get('tau_grid', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    bpr_base    = data['BPR_baseline']
    alphas      = [a for a in ALPHA_LEVELS if a in data.get('alphas', ALPHA_LEVELS)]

    results = {}

    for method in METHODS:
        if method not in data:
            print(f"    Warning: {method} not found — skipping")
            continue

        grid = tau_grid if method == 'FAIR' else lambda_grid
        results[method] = {'per_alpha': {}, 'cross_alpha': {}}

        # Accumulate Nash scores across alpha for cross-alpha consensus
        cross_nash = {}  # {param: [nash_scores across alphas]}

        for alpha_str in alphas:
            if alpha_str not in data[method]:
                continue
            if alpha_str not in bpr_base:
                continue

            a_data    = data[method][alpha_str]
            ndcg_arr  = np.array(a_data['NDCG@10']['mean'])
            aplt_arr  = np.array(a_data['APLT@10']['mean'])
            base_ndcg = float(bpr_base[alpha_str]['NDCG@10']['mean'])
            base_aplt = float(bpr_base[alpha_str]['APLT@10']['mean'])

            # Only parameters strictly inside (0, 1) for xQuAD
            # and strictly > 0 for FAIR (τ=0 = no constraint)
            n = min(len(ndcg_arr), len(grid))
            valid = [(i, grid[i]) for i in range(n)
                     if (method == 'FAIR' and grid[i] > 0.0) or
                        (method != 'FAIR' and 0.0 < grid[i] < 1.0)]

            if not valid:
                continue

            v_idx    = [i for i, _ in valid]
            v_params = [p for _, p in valid]
            v_ndcg   = ndcg_arr[v_idx]
            v_aplt   = aplt_arr[v_idx]
            v_nash   = nash_score(v_ndcg, v_aplt, base_ndcg, base_aplt)

            best_i  = int(np.argmax(v_nash))
            opt_p   = v_params[best_i]

            results[method]['per_alpha'][alpha_str] = {
                'optimal_param':   opt_p,
                'nash_score':      float(v_nash[best_i]),
                'ndcg_at_opt':     float(v_ndcg[best_i]),
                'aplt_at_opt':     float(v_aplt[best_i]),
                'ndcg_change_pct': (v_ndcg[best_i] - base_ndcg) / base_ndcg * 100,
                'aplt_change_pct': (v_aplt[best_i] - base_aplt) / base_aplt * 100,
                'baseline_ndcg':   base_ndcg,
                'baseline_aplt':   base_aplt,
                'all_params':      v_params,
                'all_nash':        v_nash.tolist(),
                'all_ndcg':        v_ndcg.tolist(),
                'all_aplt':        v_aplt.tolist(),
            }

            # Accumulate for cross-alpha
            for p, ns in zip(v_params, v_nash):
                cross_nash.setdefault(p, []).append(float(ns))

        # Cross-alpha consensus: mean Nash across alpha levels
        if cross_nash:
            mean_nash = {p: np.mean(v) for p, v in cross_nash.items()}
            best_p    = max(mean_nash, key=mean_nash.get)
            results[method]['cross_alpha'] = {
                'consensus_param': best_p,
                'mean_nash_score': mean_nash[best_p],
                'all_params':      sorted(mean_nash.keys()),
                'all_mean_nash':   [mean_nash[p]
                                    for p in sorted(mean_nash.keys())],
            }

    return results, alphas


# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────
def plot_nash_curves(results, alphas, dataset_name, save_path):
    """Nash score vs parameter for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Nash Bargaining Score — {dataset_name}',
                 fontsize=13, fontweight='bold')

    for idx, method in enumerate(['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']):
        ax = axes[idx]
        ax.grid(True, alpha=0.3)
        style = METHODS[method]

        if method not in results:
            ax.set_title(f"{style['label']} (no data)")
            continue

        for alpha_str in alphas:
            if alpha_str not in results[method]['per_alpha']:
                continue
            r     = results[method]['per_alpha'][alpha_str]
            color = ALPHA_COLORS.get(alpha_str, 'gray')

            ax.plot(r['all_params'], r['all_nash'], color=color,
                    linewidth=2, label=ALPHA_LABELS[alpha_str])
            ax.scatter(r['all_params'], r['all_nash'],
                       color=color, s=30, zorder=3)
            # Mark optimum
            oi = r['all_params'].index(r['optimal_param'])
            ax.scatter(r['optimal_param'], r['all_nash'][oi],
                       color='red', s=200, marker='*',
                       zorder=5, edgecolors='black', linewidths=1)

        # Mark cross-alpha consensus
        if 'cross_alpha' in results[method]:
            cp = results[method]['cross_alpha']['consensus_param']
            ax.axvline(cp, color='black', linestyle='--',
                       linewidth=1.5, alpha=0.7,
                       label=f'Consensus {style["param"]}={cp}')

        ax.set_xlabel(f'Parameter {style["param"]}', fontsize=11)
        ax.set_ylabel('Nash Score', fontsize=11)
        ax.set_title(style['label'], fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pareto(results, alphas, dataset_name, save_path):
    """Pareto frontier: NDCG% change vs APLT% change."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Pareto Frontier: Accuracy–Fairness Trade-off — {dataset_name}',
                 fontsize=13, fontweight='bold')

    for idx, method in enumerate(['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']):
        ax = axes[idx]
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        style = METHODS[method]

        if method not in results:
            ax.set_title(f"{style['label']} (no data)")
            continue

        legend_elements = []

        for alpha_str in alphas:
            if alpha_str not in results[method]['per_alpha']:
                continue
            r     = results[method]['per_alpha'][alpha_str]
            color = ALPHA_COLORS.get(alpha_str, 'gray')

            ndcg_pct = [(n - r['baseline_ndcg']) / r['baseline_ndcg'] * 100
                        for n in r['all_ndcg']]
            aplt_pct = [(a - r['baseline_aplt']) / r['baseline_aplt'] * 100
                        for a in r['all_aplt']]

            ax.plot(ndcg_pct, aplt_pct, color=color, linewidth=1.5, alpha=0.8)
            ax.scatter(ndcg_pct, aplt_pct, color=color, s=30, alpha=0.6)

            oi = r['all_params'].index(r['optimal_param'])
            ax.scatter(ndcg_pct[oi], aplt_pct[oi],
                       color='red', s=200, marker='*',
                       edgecolors='black', linewidths=1, zorder=5)

            legend_elements.append(
                Line2D([0], [0], color=color, lw=2,
                       label=ALPHA_LABELS[alpha_str])
            )

        ax.set_xlabel('NDCG@10 Change (%)', fontsize=11)
        ax.set_ylabel('APLT@10 Change (%)', fontsize=11)
        ax.set_title(style['label'], fontsize=12, fontweight='bold')
        if idx == 0:
            ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cross_alpha_consensus(results, dataset_name, save_path):
    """Mean Nash score vs parameter (averaged across alpha levels)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, alpha=0.3)

    for method, style in METHODS.items():
        if method not in results or 'cross_alpha' not in results[method]:
            continue
        ca = results[method]['cross_alpha']
        ax.plot(ca['all_params'], ca['all_mean_nash'],
                color=style['color'], linewidth=2.5,
                marker=style['marker'], markersize=7,
                label=style['label'])
        # Mark consensus
        cp = ca['consensus_param']
        ci = ca['all_params'].index(cp)
        ax.scatter(cp, ca['all_mean_nash'][ci],
                   color='red', s=300, marker='*',
                   zorder=5, edgecolors='black', linewidths=1)
        ax.annotate(f"{style['param']}*={cp:.1f}",
                    xy=(cp, ca['all_mean_nash'][ci]),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    color=style['color'])

    ax.set_xlabel('Parameter λ / τ', fontsize=12)
    ax.set_ylabel('Mean Nash Score (across α levels)', fontsize=12)
    ax.set_title(f'Cross-Alpha Consensus Parameter — {dataset_name}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────
def print_summary(results, alphas, dataset_name):
    print(f"\n{'=' * 65}")
    print(f"  {dataset_name} — PER-ALPHA OPTIMAL PARAMETERS")
    print(f"{'=' * 65}")
    print(f"  {'Method':>14} {'α':>6} {'λ*/τ*':>6} "
          f"{'NDCG Δ%':>9} {'APLT Δ%':>9} {'Nash':>7}")
    print(f"  {'-' * 55}")

    for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
        if method not in results:
            continue
        for alpha_str in alphas:
            if alpha_str not in results[method]['per_alpha']:
                continue
            r = results[method]['per_alpha'][alpha_str]
            print(f"  {METHODS[method]['label']:>14} {alpha_str:>6} "
                  f"{r['optimal_param']:>6.1f} "
                  f"{r['ndcg_change_pct']:>+9.1f}% "
                  f"{r['aplt_change_pct']:>+9.1f}% "
                  f"{r['nash_score']:>7.3f}")
        print()

    print(f"\n{'=' * 65}")
    print(f"  {dataset_name} — CROSS-ALPHA CONSENSUS (for LightGCN / NeuMF)")
    print(f"{'=' * 65}")
    print(f"  {'Method':>14} {'Consensus λ*/τ*':>16} {'Mean Nash':>12}")
    print(f"  {'-' * 45}")
    for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
        if method not in results or 'cross_alpha' not in results[method]:
            continue
        ca = results[method]['cross_alpha']
        print(f"  {METHODS[method]['label']:>14} "
              f"{ca['consensus_param']:>16.1f} "
              f"{ca['mean_nash_score']:>12.4f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  NASH BARGAINING PARAMETER SELECTION")
    print("  Nash(p) = (NDCG / NDCG_base) × (APLT / APLT_base)")
    print("=" * 65)

    all_consensus = {}   # {dataset: {method: param}}
    combined_rows = []

    for dataset_name, paths in DATASETS.items():
        agg_path = paths['aggregated']
        out_dir  = paths['output']
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(agg_path):
            print(f"\n  SKIP {dataset_name}: {agg_path} not found")
            continue

        print(f"\nLoading {dataset_name}: {agg_path}")
        with open(agg_path) as f:
            data = json.load(f)

        results, alphas = compute_nash(data)

        print_summary(results, alphas, dataset_name)

        # Save per-alpha CSV
        rows = []
        for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
            if method not in results:
                continue
            for alpha_str in alphas:
                if alpha_str not in results[method]['per_alpha']:
                    continue
                r = results[method]['per_alpha'][alpha_str]
                row = {
                    'Dataset':       dataset_name,
                    'Method':        METHODS[method]['label'],
                    'Alpha':         float(alpha_str),
                    'Optimal_Param': r['optimal_param'],
                    'NDCG_Change_%': f"{r['ndcg_change_pct']:+.1f}",
                    'APLT_Change_%': f"{r['aplt_change_pct']:+.1f}",
                    'Nash_Score':    round(r['nash_score'], 4),
                }
                rows.append(row)
                combined_rows.append(row)

        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, 'nash_per_alpha.csv'), index=False
        )

        # Save cross-alpha consensus
        consensus = {}
        all_consensus[dataset_name] = {}
        for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
            if method not in results or 'cross_alpha' not in results[method]:
                continue
            ca = results[method]['cross_alpha']
            label = METHODS[method]['label']
            consensus[label] = {
                'consensus_param': ca['consensus_param'],
                'mean_nash_score': round(ca['mean_nash_score'], 4),
            }
            all_consensus[dataset_name][method] = ca['consensus_param']

        with open(os.path.join(out_dir, 'consensus_params.json'), 'w') as f:
            json.dump(consensus, f, indent=2)

        # Figures
        print(f"\n  Generating figures for {dataset_name} ...")
        plot_nash_curves(
            results, alphas, dataset_name,
            os.path.join(out_dir, 'nash_curves.png')
        )
        plot_pareto(
            results, alphas, dataset_name,
            os.path.join(out_dir, 'pareto_frontiers.png')
        )
        plot_cross_alpha_consensus(
            results, dataset_name,
            os.path.join(out_dir, 'consensus_curve.png')
        )

    # ── Cross-dataset consensus table ─────────────────────────
    print(f"\n{'=' * 65}")
    print("  FINAL CONSENSUS PARAMETERS FOR CROSS-MODEL EXPERIMENT")
    print("  (Apply these unchanged to LightGCN and NeuMF)")
    print(f"{'=' * 65}")
    print(f"  {'Method':>14} {'ML param':>10} {'AB param':>10} "
          f"{'Use':>8}")
    print(f"  {'-' * 45}")

    final_params = {}
    for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
        label = METHODS[method]['label']
        ml_p  = all_consensus.get('MovieLens', {}).get(method, '-')
        ab_p  = all_consensus.get('AmazonBooks', {}).get(method, '-')
        # If both agree, use that; otherwise use MovieLens (larger experiment)
        use_p = ml_p if ml_p == ab_p else ml_p
        final_params[label] = use_p
        match = '✓ agree' if ml_p == ab_p else '← ML'
        print(f"  {label:>14} {str(ml_p):>10} {str(ab_p):>10} "
              f"{match:>8}")

    # Save final params
    for dataset_name, paths in DATASETS.items():
        out_dir = paths['output']
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'final_cross_model_params.json'), 'w') as f:
            json.dump(final_params, f, indent=2)

    # Combined CSV
    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(
            'nash_combined_all_datasets.csv', index=False
        )

    print(f"\n{'=' * 65}")
    print("  Nash selection complete.")
    print("  Use consensus_params.json as fixed parameters for")
    print("  LightGCN and NeuMF experiments.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()