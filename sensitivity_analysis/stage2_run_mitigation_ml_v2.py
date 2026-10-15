"""
STAGE 2 — Mitigation Sweep on Saved Candidates (MovieLens)
===========================================================
Key fixes vs original:
  1. SEEDS and PATHS updated to v1
  2. Direct evaluation from eval_bundle ground_truth
  3. FA*IR uses corrected threshold (see fair_reranker.py)
  4. User data precomputed ONCE per (seed, alpha)
  5. TAU_GRID excludes 0.0
  6. BPR_baseline per-user serialization fixed (flat dict,
     no lambda nesting — was crashing with AttributeError)
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aggregation import aggregate_across_seeds, compute_sweet_spot
from reranking.xquad_binary import BinaryxQuAD
from reranking.xquad_smooth import SmoothxQuAD
from reranking.fair_reranker_v1 import FAIRReranker

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
CANDIDATES_DIR = r"D:\Final_thesis\thesis_project\sensitivity_analysis\candidates_ml_v2"
OUTPUT_DIR     = r"D:\Final_thesis\thesis_project\sensitivity_analysis\sa_results_ml_v2"

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SEEDS        = [42]
ALPHA_LEVELS = [0.5, 0.75]
K            = 10
RERANK_FROM  = 100
LAMBDA_GRID  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAU_GRID     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ensure_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    return obj


def serialize_per_user(per_user: dict) -> dict:
    """
    Serialize per-user metrics to JSON-compatible format.

    Structure of per_user:
      {alpha_val: {
          "BPR_baseline": {user_id: ndcg_val},          ← flat, no lambda
          "BinaryxQuAD":  {lambda: {user_id: ndcg_val}}, ← lambda-nested
          "SmoothxQuAD":  {lambda: {user_id: ndcg_val}},
          "FAIR":         {tau:    {user_id: ndcg_val}},
      }}

    BPR_baseline has NO lambda dimension — treating it the same as
    rerankers causes 'numpy.float64 has no .items()' crash.
    """
    result = {}
    for alpha, alpha_dict in per_user.items():
        result[str(alpha)] = {}
        for algo, algo_dict in alpha_dict.items():
            if algo == "BPR_baseline":
                # Flat {user_id: ndcg} — just convert keys to str
                result[str(alpha)][algo] = {
                    str(u): v for u, v in algo_dict.items()
                }
            else:
                # {lambda: {user_id: ndcg}} — two levels
                result[str(alpha)][algo] = {
                    str(lam): {str(u): v for u, v in ud.items()}
                    for lam, ud in algo_dict.items()
                }
    return ensure_serializable(result)


def deserialize_per_user(per_user_json: dict) -> dict:
    """
    Reverse of serialize_per_user — used when loading skipped seeds.
    Restores float alpha keys, correct types for lambda and user keys.
    """
    result = {}
    for alpha_str, alpha_dict in per_user_json.items():
        alpha_f = float(alpha_str)
        result[alpha_f] = {}
        for algo_name, algo_dict in alpha_dict.items():
            if algo_name == "BPR_baseline":
                result[alpha_f][algo_name] = {
                    int(u): v for u, v in algo_dict.items()
                }
            else:
                result[alpha_f][algo_name] = {
                    float(lam): {int(u): v for u, v in ud.items()}
                    for lam, ud in algo_dict.items()
                }
    return result


def is_seed_complete(seed_dir: str) -> bool:
    required = ["results.csv", "nested.json", "per_user_metrics.json"]
    return os.path.isdir(seed_dir) and all(
        os.path.exists(os.path.join(seed_dir, f)) for f in required
    )


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_seed_alpha(seed: int, alpha_val: float) -> dict:
    save_dir = os.path.join(
        CANDIDATES_DIR, f"seed_{seed}", f"alpha_{alpha_val}"
    )
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Candidates not found: {save_dir}")

    with open(os.path.join(save_dir, "candidates.pkl"),    "rb") as f:
        candidates = pickle.load(f)
    with open(os.path.join(save_dir, "user_profiles.pkl"), "rb") as f:
        user_profiles = pickle.load(f)
    with open(os.path.join(save_dir, "eval_bundle.pkl"),   "rb") as f:
        eval_bundle = pickle.load(f)

    if isinstance(eval_bundle.get('long_tail_items'), list):
        eval_bundle['long_tail_items']  = set(eval_bundle['long_tail_items'])
    if isinstance(eval_bundle.get('short_head_items'), list):
        eval_bundle['short_head_items'] = set(eval_bundle['short_head_items'])

    if 'ground_truth' in eval_bundle:
        eval_bundle['ground_truth'] = {
            int(uid): set(items)
            for uid, items in eval_bundle['ground_truth'].items()
        }

    return {
        "candidates":    candidates,
        "user_profiles": user_profiles,
        "eval_bundle":   eval_bundle,
    }


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(recs: dict, ground_truth: dict,
             item_popularity: dict, long_tail_items: set,
             k: int = K) -> tuple:
    """
    Compute aggregate metrics and per-user NDCG directly from
    eval_bundle contents. No EvaluationRunner needed.
    """
    ndcg_scores, recall_scores, aplt_scores = [], [], []
    per_user_ndcg, per_user_aplt = {}, {}
    total_pop, n_recs_total = 0.0, 0

    for user, rec_list in recs.items():
        true_items = ground_truth.get(user, set())
        if not true_items:
            continue
        predicted = [item for item, _ in rec_list[:k]]
        if not predicted:
            continue

        relevances = [1 if item in true_items else 0 for item in predicted]
        dcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
        idcg = sum(1.0 / np.log2(i + 2)
                   for i in range(min(len(true_items), k)))
        ndcg_val   = dcg / idcg if idcg > 0 else 0.0
        recall_val = sum(1 for item in predicted
                         if item in true_items) / len(true_items)
        n_lt       = sum(1 for item in predicted if item in long_tail_items)
        aplt_val   = n_lt / len(predicted)

        for item in predicted:
            total_pop    += item_popularity.get(item, 0)
            n_recs_total += 1

        ndcg_scores.append(ndcg_val)
        recall_scores.append(recall_val)
        aplt_scores.append(aplt_val)
        per_user_ndcg[user] = ndcg_val
        per_user_aplt[user] = aplt_val

    arp = total_pop / n_recs_total if n_recs_total > 0 else 0.0
    aggregate = {
        f'NDCG@{k}':   float(np.mean(ndcg_scores))   if ndcg_scores   else 0.0,
        f'Recall@{k}': float(np.mean(recall_scores))  if recall_scores else 0.0,
        f'APLT@{k}':   float(np.mean(aplt_scores))    if aplt_scores   else 0.0,
        f'ARP@{k}':    float(arp),
    }
    return aggregate, per_user_ndcg, per_user_aplt


# ─────────────────────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────────────────────
def run_xquad(reranker_cls, user_data: dict,
              long_tail_items: set, short_head_items: set,
              lambda_val: float) -> dict:
    reranker = reranker_cls(
        lambda_param=lambda_val, rerank_from=RERANK_FROM, rerank_to=K
    )
    recs = {}
    for user, ud in user_data.items():
        recs[user] = reranker.rerank(
            user_idx        = user,
            candidate_items = ud['items'],
            base_scores     = ud['norm_scores'],
            user_profile    = ud['profile'],
            long_tail_items = long_tail_items,
            short_head_items= short_head_items,
        )
    return recs


def run_fair(user_data: dict, long_tail_items: set, tau: float) -> dict:
    reranker = FAIRReranker(p=tau, alpha=0.05, k=K)
    recs     = {}
    for user, ud in user_data.items():
        protected_mask = {item: (item in long_tail_items)
                          for item in ud['items']}
        scores         = ud['scores']
        reranked       = reranker.rerank(ud['items'], scores, protected_mask)
        recs[user]     = [(item, scores.get(item, 0.0)) for item in reranked]
    return recs


# ─────────────────────────────────────────────────────────────
# SINGLE-SEED SWEEP
# ─────────────────────────────────────────────────────────────
def run_single_seed(seed: int, alpha_vals: list) -> tuple:
    all_rows     = []
    per_user_out = {}
    nested_out   = {}

    for alpha_val in alpha_vals:
        alpha_str = str(alpha_val)
        print(f"    α={alpha_val} ...", end=" ", flush=True)

        try:
            saved = load_seed_alpha(seed, alpha_val)
        except FileNotFoundError as e:
            print(f"\n    ERROR: {e}")
            continue

        candidates      = saved["candidates"]
        user_profiles   = saved["user_profiles"]
        bundle          = saved["eval_bundle"]
        ground_truth    = bundle["ground_truth"]
        item_popularity = bundle["item_popularity"]
        long_tail_items = bundle["long_tail_items"]
        short_head_items= bundle["short_head_items"]

        # Precompute user data ONCE — reused across all lambda values
        user_data = {}
        for user, item_list in candidates.items():
            items  = [item for item, _ in item_list[:RERANK_FROM]]
            scores = {item: score for item, score in item_list[:RERANK_FROM]}
            max_s  = max(scores.values()) if scores else 1.0
            norm   = {k: v / max_s for k, v in scores.items()} \
                     if max_s > 0 else scores
            user_data[user] = {
                'items':       items,
                'scores':      scores,
                'norm_scores': norm,
                'profile':     user_profiles.get(user, set()),
            }

        nested_out[alpha_str]   = {
            "binary_xquad": {}, "smooth_xquad": {}, "fair": {}
        }
        per_user_out[alpha_val] = {
            "BPR_baseline": {}, "BinaryxQuAD": {},
            "SmoothxQuAD":  {}, "FAIR": {}
        }

        # BPR Baseline
        baseline_recs = {
            u: [(item, user_data[u]['scores'].get(item, 0.0))
                for item in user_data[u]['items'][:K]]
            for u in user_data
        }
        bl_agg, bl_ndcg, _ = evaluate(
            baseline_recs, ground_truth, item_popularity, long_tail_items
        )
        per_user_out[alpha_val]["BPR_baseline"] = bl_ndcg
        all_rows.append({
            "seed": seed, "alpha": alpha_val,
            "algorithm": "BPR_baseline", "lambda": None, **bl_agg
        })

        # Binary xQuAD
        for lam in LAMBDA_GRID:
            recs = run_xquad(BinaryxQuAD, user_data,
                             long_tail_items, short_head_items, lam)
            agg, pu_ndcg, _ = evaluate(
                recs, ground_truth, item_popularity, long_tail_items
            )
            nested_out[alpha_str]["binary_xquad"][str(lam)] = agg
            per_user_out[alpha_val]["BinaryxQuAD"][lam]     = pu_ndcg
            all_rows.append({
                "seed": seed, "alpha": alpha_val,
                "algorithm": "BinaryxQuAD", "lambda": lam, **agg
            })

        # Smooth xQuAD
        for lam in LAMBDA_GRID:
            recs = run_xquad(SmoothxQuAD, user_data,
                             long_tail_items, short_head_items, lam)
            agg, pu_ndcg, _ = evaluate(
                recs, ground_truth, item_popularity, long_tail_items
            )
            nested_out[alpha_str]["smooth_xquad"][str(lam)] = agg
            per_user_out[alpha_val]["SmoothxQuAD"][lam]     = pu_ndcg
            all_rows.append({
                "seed": seed, "alpha": alpha_val,
                "algorithm": "SmoothxQuAD", "lambda": lam, **agg
            })

        # FA*IR
        for tau in TAU_GRID:
            recs = run_fair(user_data, long_tail_items, tau)
            agg, pu_ndcg, _ = evaluate(
                recs, ground_truth, item_popularity, long_tail_items
            )
            nested_out[alpha_str]["fair"][str(tau)] = agg
            per_user_out[alpha_val]["FAIR"][tau]    = pu_ndcg
            all_rows.append({
                "seed": seed, "alpha": alpha_val,
                "algorithm": "FAIR", "lambda": tau, **agg
            })

        print("done")

    return pd.DataFrame(all_rows), per_user_out, nested_out


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_stage2():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 65}")
    print(f"  STAGE 2 — Mitigation Sweep")
    print(f"  Timestamp    : {timestamp}")
    print(f"  Seeds        : {SEEDS}")
    print(f"  Alpha levels : {ALPHA_LEVELS}")
    print(f"  Lambda grid  : {LAMBDA_GRID}")
    print(f"  Tau grid     : {TAU_GRID}")
    print(f"  Candidates   : {CANDIDATES_DIR}")
    print(f"{'=' * 65}")

    all_results     = {}
    all_per_user    = {}
    completed_seeds = []
    skipped_seeds   = []

    for seed in SEEDS:
        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")

        if is_seed_complete(seed_dir):
            print(f"\n{'─'*55}  seed={seed} [SKIP — already complete]")
            skipped_seeds.append(seed)
            try:
                results_df = pd.read_csv(
                    os.path.join(seed_dir, "results.csv")
                )
                with open(os.path.join(seed_dir, "nested.json")) as f:
                    nested = json.load(f)
                with open(os.path.join(seed_dir,
                                       "per_user_metrics.json")) as f:
                    per_user_raw = json.load(f)

                # Restore correct types from JSON string keys
                per_user = deserialize_per_user(per_user_raw)

                all_results[seed]  = {"results_df": results_df,
                                       "nested":     nested,
                                       "per_user":   per_user}
                all_per_user[seed] = per_user
            except Exception as e:
                print(f"    WARNING: Could not load existing: {e}")
            continue

        print(f"\n{'─'*55}  seed={seed}")
        try:
            results_df, per_user, nested = run_single_seed(
                seed, ALPHA_LEVELS
            )
            all_results[seed]  = {"results_df": results_df,
                                   "nested":     nested,
                                   "per_user":   per_user}
            all_per_user[seed] = per_user
            completed_seeds.append(seed)

            os.makedirs(seed_dir, exist_ok=True)
            results_df.to_csv(
                os.path.join(seed_dir, "results.csv"), index=False
            )

            with open(os.path.join(seed_dir, "nested.json"), "w") as f:
                json.dump(ensure_serializable(nested), f, indent=2)

            # KEY FIX: use serialize_per_user which handles the
            # BPR_baseline flat structure separately from rerankers
            with open(os.path.join(seed_dir,
                                   "per_user_metrics.json"), "w") as f:
                json.dump(serialize_per_user(per_user), f, indent=2)

        except Exception as e:
            print(f"    ERROR processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 65}")
    print(f"  Completed: {completed_seeds}")
    print(f"  Skipped  : {skipped_seeds}")
    print(f"{'=' * 65}")

    if not all_results:
        print("  No results to aggregate.")
        return

    print("\n  Aggregating across seeds ...")
    enhanced_dir = os.path.join(OUTPUT_DIR, "enhanced_analysis")
    os.makedirs(enhanced_dir, exist_ok=True)

    aggregated = aggregate_across_seeds(all_results, LAMBDA_GRID, TAU_GRID)
    with open(os.path.join(enhanced_dir,
                            "aggregated_results.json"), "w") as f:
        json.dump(ensure_serializable(aggregated), f, indent=2)

    all_df = pd.concat(
        [all_results[s]["results_df"] for s in all_results],
        ignore_index=True
    )
    all_df.to_csv(
        os.path.join(enhanced_dir, "all_results_combined.csv"), index=False
    )

    summary = all_df.groupby(['alpha', 'algorithm']).agg(
        NDCG_mean  =('NDCG@10',   'mean'), NDCG_std  =('NDCG@10',   'std'),
        APLT_mean  =('APLT@10',   'mean'), APLT_std  =('APLT@10',   'std'),
        ARP_mean   =('ARP@10',    'mean'), ARP_std   =('ARP@10',    'std'),
        Recall_mean=('Recall@10', 'mean'), Recall_std=('Recall@10', 'std'),
    ).round(4)
    summary.to_csv(os.path.join(enhanced_dir, "summary_table.csv"))

    print(f"\n  Quick summary (best λ/τ per algo, α=1.0 and α=2.0):")
    print(f"  {'α':>5} {'Algorithm':>14} {'NDCG@10':>10} {'APLT@10':>10}")
    print(f"  {'-'*45}")
    for alpha_val in [1.0, 2.0]:
        for algo in ['BPR_baseline', 'BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
            rows = all_df[
                (all_df['alpha'] == alpha_val) &
                (all_df['algorithm'] == algo)
            ]
            if rows.empty:
                continue
            best = rows.loc[rows['APLT@10'].idxmax()]
            print(f"  {alpha_val:>5} {algo:>14} "
                  f"{best['NDCG@10']:>10.4f} "
                  f"{best['APLT@10']:>10.4f}")

    print(f"\n{'=' * 65}")
    print(f"  STAGE 2 COMPLETE")
    print(f"  Results → {OUTPUT_DIR}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_stage2()