"""
Utility functions for aggregating results across multiple seeds.
"""

import numpy as np
from collections import defaultdict


def aggregate_across_seeds(all_results: dict, lambda_grid: list, tau_grid: list) -> dict:
    """
    Aggregate results across multiple seeds with mean and std.
    Returns a JSON-serializable dictionary (no numpy arrays, no circular refs).
    """
    aggregated = {
        'alphas': [],
        'lambda_grid': [float(x) for x in lambda_grid],
        'tau_grid': [float(x) for x in tau_grid],
        'BinaryxQuAD': {},
        'SmoothxQuAD': {},
        'FAIR': {},
        'BPR_baseline': {},
    }

    first_seed = list(all_results.keys())[0]
    alphas = sorted(all_results[first_seed]['nested'].keys(), key=float)
    aggregated['alphas'] = alphas

    algo_map = [
        ('BinaryxQuAD', 'binary_xquad'),
        ('SmoothxQuAD', 'smooth_xquad'),
        ('FAIR', 'fair'),
    ]

    for alpha_str in alphas:

        # ── Re-ranker algorithms ──────────────────────────────────────────
        for algo, algo_key in algo_map:
            raw = defaultdict(lambda: defaultdict(list))

            for seed, seed_results in all_results.items():
                if alpha_str not in seed_results['nested']:
                    continue
                seed_algo = seed_results['nested'][alpha_str].get(algo_key, {})
                for param_str, metrics in seed_algo.items():
                    for metric_name, value in metrics.items():
                        raw[param_str][metric_name].append(float(value))

            sorted_params = sorted(raw.keys(), key=float)

            all_metric_names = set()
            for p in sorted_params:
                all_metric_names.update(raw[p].keys())

            algo_alpha = {}
            for metric_name in all_metric_names:
                means, stds, mins, maxs = [], [], [], []
                for param_str in sorted_params:
                    vals = raw[param_str].get(metric_name, [0.0])
                    means.append(float(np.mean(vals)) if vals else 0.0)
                    stds.append(float(np.std(vals)) if vals else 0.0)
                    mins.append(float(np.min(vals)) if vals else 0.0)
                    maxs.append(float(np.max(vals)) if vals else 0.0)

                algo_alpha[metric_name] = {
                    'mean': means,
                    'std': stds,
                    'min': mins,
                    'max': maxs,
                }

            aggregated[algo][alpha_str] = algo_alpha

        # ── Baseline (scalar per metric) ─────────────────────────────────
        baseline_metrics = defaultdict(list)
        for seed, seed_results in all_results.items():
            df = seed_results['results_df']
            rows = df[
                (df['alpha'] == float(alpha_str)) &
                (df['algorithm'] == 'BPR_baseline')
                ]
            if rows.empty:
                continue
            for metric in ['NDCG@10', 'APLT@10', 'ARP@10', 'Recall@10', 'ACLT@10', 'AggregateDiversity@10']:
                if metric in rows.columns:
                    baseline_metrics[metric].append(float(rows[metric].iloc[0]))

        aggregated['BPR_baseline'][alpha_str] = {
            metric: {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }
            for metric, vals in baseline_metrics.items() if vals
        }

    return aggregated


def compute_sweet_spot(
        ndcg_vals: list,
        aplt_vals: list,
        param_grid: list,
        threshold: float = 0.9,
) -> dict:
    """
    Identify the sweet-spot region where both NDCG and APLT are within
    `threshold` × their respective maxima.
    """
    ndcg_arr = np.array(ndcg_vals)
    aplt_arr = np.array(aplt_vals)

    ndcg_max = ndcg_arr.max()
    aplt_max = aplt_arr.max()

    ndcg_thresh = threshold * ndcg_max
    aplt_thresh = threshold * aplt_max

    good = np.where((ndcg_arr >= ndcg_thresh) & (aplt_arr >= aplt_thresh))[0]
    if good.size == 0:
        good = np.where((ndcg_arr >= ndcg_thresh) | (aplt_arr >= aplt_thresh))[0]

    if good.size > 0:
        regions = []
        start = good[0]
        for i in range(1, len(good)):
            if good[i] > good[i - 1] + 1:
                regions.append((start, good[i - 1]))
                start = good[i]
        regions.append((start, good[-1]))

        ndcg_range = ndcg_max - ndcg_arr.min() + 1e-10
        aplt_range = aplt_max - aplt_arr.min() + 1e-10

        best_region, best_score = None, -np.inf
        for s, e in regions:
            ndcg_norm = (ndcg_arr[s:e + 1] - ndcg_arr.min()) / ndcg_range
            aplt_norm = (aplt_arr[s:e + 1] - aplt_arr.min()) / aplt_range
            score = ((ndcg_norm + aplt_norm) / 2).mean()
            if score > best_score:
                best_score, best_region = score, (s, e)

        if best_region:
            s, e = best_region
            return {
                'start_param': float(param_grid[s]),
                'end_param': float(param_grid[e]),
                'start_idx': int(s),
                'end_idx': int(e),
                'ndcg_range': [float(ndcg_arr[s]), float(ndcg_arr[e])],
                'aplt_range': [float(aplt_arr[s]), float(aplt_arr[e])],
                'composite_score': float(best_score),
            }

    # Fallback: single best composite point
    ndcg_norm = (ndcg_arr - ndcg_arr.min()) / (ndcg_max - ndcg_arr.min() + 1e-10)
    aplt_norm = (aplt_arr - aplt_arr.min()) / (aplt_max - aplt_arr.min() + 1e-10)
    composite = (ndcg_norm + aplt_norm) / 2
    best_idx = int(np.argmax(composite))

    return {
        'start_param': float(param_grid[best_idx]),
        'end_param': float(param_grid[best_idx]),
        'start_idx': best_idx,
        'end_idx': best_idx,
        'ndcg_range': [float(ndcg_arr[best_idx]), float(ndcg_arr[best_idx])],
        'aplt_range': [float(aplt_arr[best_idx]), float(aplt_arr[best_idx])],
        'composite_score': float(composite[best_idx]),
    }