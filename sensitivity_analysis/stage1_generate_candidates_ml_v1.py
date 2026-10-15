# stage1_generate_candidates_ml.py
import os
import json
import pickle
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from model_training.bpr_trainer_v1 import BPRTrainer
from evaluation.evaluation_v1 import RecommenderMetrics

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_DIR             = r"D:\Final_thesis\thesis_project\data_preparation\biased_datasets\MovieLens_v1"
CANDIDATES_DIR       = r"D:\Final_thesis\thesis_project\sensitivity_analysis\candidates_ml_v1"
BASELINE_METRICS_DIR = r"D:\Final_thesis\thesis_project\sensitivity_analysis\baseline_metrics_ml_v1"

# ─────────────────────────────────────────────────────────────
# CONSTANTS — must match bias_induction and stage2
# ─────────────────────────────────────────────────────────────
SEEDS        = [42, 123, 456]                          # 3 seeds
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]             # 5 levels
RERANK_FROM  = 100
EPOCHS       = 50
EMBEDDING_DIM = 64
LR           = 0.02
BATCH_SIZE   = 2048
DEVICE       = "cuda"
K            = 10                                       # single cutoff


# ─────────────────────────────────────────────────────────────
# FIXED LONG-TAIL LOADING
# Replaces build_popularity_sets entirely.
# Loaded ONCE before all loops.
# ─────────────────────────────────────────────────────────────
def load_fixed_long_tail(data_dir: str):
    """
    Load fixed long-tail definition from bias induction output.
    Returns original item IDs (not internal indices).
    IDs are converted to int for MovieLens (int IDs) and kept as
    str for Amazon Books (ASIN strings).
    Remapping to trainer's internal indices happens after training.
    """
    path = os.path.join(data_dir, 'fixed_long_tail.json')
    with open(path) as f:
        fixed = json.load(f)

    def convert(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return x

    tail_ids = {convert(x) for x in fixed['tail_item_ids']}
    head_ids = {convert(x) for x in fixed['head_item_ids']}

    print(f"  Fixed long-tail loaded: "
          f"{len(head_ids):,} head / {len(tail_ids):,} tail items")
    return tail_ids, head_ids


def load_original_item_popularity(data_dir: str):
    """
    Load item popularity counts from original filtered data.
    Used for ARP — fixed reference across all alpha levels.
    """
    path = os.path.join(data_dir, 'item_pop_original.json')
    with open(path) as f:
        raw = json.load(f)

    def convert(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return x

    return {convert(k): int(v) for k, v in raw.items()}


def remap_fixed_long_tail(tail_ids, head_ids, item_pop_original,
                           trainer_item_map):
    """
    Remap fixed original IDs to trainer's internal indices.
    Items absent from trainer_item_map had zero training
    interactions at this alpha level — structural depletion.
    """
    tail_int = {trainer_item_map[iid]
                for iid in tail_ids if iid in trainer_item_map}
    head_int = {trainer_item_map[iid]
                for iid in head_ids if iid in trainer_item_map}
    pop_int  = {trainer_item_map[iid]: cnt
                for iid, cnt in item_pop_original.items()
                if iid in trainer_item_map}

    n_missing = len(tail_ids) - len(tail_int)
    return tail_int, head_int, pop_int, n_missing


# ─────────────────────────────────────────────────────────────
# CANDIDATE POOL ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyse_candidate_pool(candidates: dict,
                            long_tail_int: set,
                            pool_size: int = 100) -> dict:
    """
    For each user, measure how many genuine tail items appear
    in the top-pool_size candidates.
    This quantifies the structural ceiling on APLT improvement.
    """
    per_user_lt_in_pool = []
    users_zero_lt = 0

    for user, cand_list in candidates.items():
        lt_count = sum(1 for item, _ in cand_list[:pool_size]
                       if item in long_tail_int)
        per_user_lt_in_pool.append(lt_count / pool_size)
        if lt_count == 0:
            users_zero_lt += 1

    return {
        'mean_lt_fraction_in_pool':  float(np.mean(per_user_lt_in_pool)),
        'median_lt_fraction_in_pool': float(np.median(per_user_lt_in_pool)),
        'users_with_zero_lt_in_pool': users_zero_lt,
        'pct_users_zero_lt':         users_zero_lt / len(candidates) * 100,
        'theoretical_max_aplt':      float(np.mean(
            [min(1.0, f) for f in per_user_lt_in_pool]
        )),
    }


# ─────────────────────────────────────────────────────────────
# CANDIDATE GENERATION — numpy optimised
# ─────────────────────────────────────────────────────────────
def generate_candidate_pool(trainer, test_users,
                             pool_size: int = 100) -> dict:
    """
    Generate top-pool_size candidates per user.
    Uses numpy argpartition (O(n)) instead of full sort (O(n log n)).
    """
    valid_users = [u for u in test_users if u in trainer.user_map]
    user_indices = torch.LongTensor(
        [trainer.user_map[u] for u in valid_users]
    ).to(trainer.device)
    all_item_t = torch.arange(len(trainer.item_map)).to(trainer.device)

    scores_np = trainer._batch_predict(user_indices, all_item_t)

    candidates = {}
    n_items = len(trainer.item_map)

    for row_idx, user_id in enumerate(valid_users):
        user_scores = scores_np[row_idx].copy()
        pos_set = trainer.user_pos_items.get(
            trainer.user_map[user_id], set()
        )

        # Mask training items
        if pos_set:
            user_scores[np.array(list(pos_set), dtype=np.int64)] = -np.inf

        # argpartition: O(n) to get top-k, then sort just those k
        if n_items <= pool_size:
            top_idx = np.argsort(user_scores)[::-1]
        else:
            top_idx = np.argpartition(user_scores, -pool_size)[-pool_size:]
            top_idx = top_idx[np.argsort(user_scores[top_idx])[::-1]]

        candidates[trainer.user_map[user_id]] = [
            (int(idx), float(user_scores[idx])) for idx in top_idx
        ]

    return candidates


# ─────────────────────────────────────────────────────────────
# BASELINE EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate_baseline(candidates: dict, ground_truth: dict,
                       item_popularity: dict,
                       long_tail_items: set, k: int = 10) -> dict:
    """Evaluate BPR baseline before any mitigation."""
    ndcg_scores, recall_scores, per_user_ndcg, per_user_aplt = [], [], {}, {}

    for user_id, recs in candidates.items():
        true_items = ground_truth.get(user_id, set())
        if not true_items:
            continue

        predicted = [item for item, _ in recs[:k]]

        ndcg = RecommenderMetrics.ndcg_at_k(true_items, predicted, k)
        recall = RecommenderMetrics.recall_at_k(true_items, predicted, k)
        n_lt = sum(1 for item in predicted if item in long_tail_items)
        aplt = n_lt / len(predicted) if predicted else 0.0

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        per_user_ndcg[user_id] = ndcg
        per_user_aplt[user_id] = aplt

    # ARP
    total_pop, n_recs = 0.0, 0
    for user_id, recs in candidates.items():
        for item, _ in recs[:k]:
            total_pop += item_popularity.get(item, 0)
            n_recs += 1
    arp = total_pop / n_recs if n_recs > 0 else 0.0

    return {
        'aggregate': {
            f'NDCG@{k}':   float(np.mean(ndcg_scores))  if ndcg_scores  else 0.0,
            f'Recall@{k}': float(np.mean(recall_scores)) if recall_scores else 0.0,
            f'APLT@{k}':   float(np.mean(list(per_user_aplt.values()))),
            f'ARP@{k}':    float(arp),
        },
        'per_user_ndcg': per_user_ndcg,
        'per_user_aplt': per_user_aplt,
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ground_truth(test_df_int: pd.DataFrame) -> dict:
    """
    Vectorised ground truth construction.
    Replaces iterrows() which is very slow on large DataFrames.
    """
    return (
        test_df_int.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def build_user_profiles(train_df_int: pd.DataFrame) -> dict:
    """
    Vectorised user profile construction.
    Maps internal user_idx → set of internal item_idx in training.
    """
    return (
        train_df_int.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )


def remap_df(df: pd.DataFrame, user_map: dict,
             item_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)
    df.dropna(subset=["user_idx", "item_idx"], inplace=True)
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_stage1():
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    os.makedirs(BASELINE_METRICS_DIR, exist_ok=True)

    # ── Load fixed definitions ONCE ───────────────────────────
    # These never change across alpha levels or seeds
    tail_ids_fixed, head_ids_fixed = load_fixed_long_tail(DATA_DIR)
    item_pop_original = load_original_item_popularity(DATA_DIR)

    total_combos = len(ALPHA_LEVELS) * len(SEEDS)
    done = 0
    all_baseline_results = []

    print(f"\n{'=' * 65}")
    print(f"  STAGE 1 — BPR Training + Candidate Generation")
    print(f"  Alpha levels : {ALPHA_LEVELS}")
    print(f"  Seeds        : {SEEDS}")
    print(f"  Total runs   : {total_combos}")
    print(f"{'=' * 65}")

    # ── OUTER LOOP: alpha levels ──────────────────────────────
    # Each alpha's CSV is loaded ONCE and reused across all seeds
    for alpha_val in ALPHA_LEVELS:
        alpha_dir = os.path.join(DATA_DIR, f"alpha_{alpha_val}")

        if not os.path.exists(os.path.join(alpha_dir, "train.csv")):
            print(f"\n  ERROR: Missing train.csv in {alpha_dir}")
            continue

        print(f"\n{'─' * 65}")
        print(f"  α = {alpha_val}  — loading dataset")

        # Load CSV once for this alpha level
        train_df = pd.read_csv(os.path.join(alpha_dir, "train.csv"))
        test_df  = pd.read_csv(os.path.join(alpha_dir, "test.csv"))
        print(f"    Train: {len(train_df):,}  Test: {len(test_df):,}  "
              f"Test users: {test_df['user_id'].nunique():,}")

        # ── INNER LOOP: seeds ─────────────────────────────────
        for seed in SEEDS:
            done += 1
            save_dir          = os.path.join(CANDIDATES_DIR,
                                              f"seed_{seed}",
                                              f"alpha_{alpha_val}")
            baseline_dir      = os.path.join(BASELINE_METRICS_DIR,
                                              f"seed_{seed}",
                                              f"alpha_{alpha_val}")

            # ── Resume: skip if already complete ─────────────
            required_files = ["candidates.pkl", "user_profiles.pkl",
                              "eval_bundle.pkl", "meta.json"]
            if (all(os.path.exists(os.path.join(save_dir, f))
                    for f in required_files) and
                os.path.exists(os.path.join(baseline_dir,
                                            "baseline_metrics.json"))):
                print(f"  [{done}/{total_combos}] SKIP  "
                      f"seed={seed} α={alpha_val}")
                try:
                    with open(os.path.join(baseline_dir,
                                           "baseline_metrics.json")) as f:
                        bd = json.load(f)
                        bd['seed']  = seed
                        bd['alpha'] = alpha_val
                        all_baseline_results.append(bd)
                except Exception:
                    pass
                continue

            os.makedirs(save_dir,     exist_ok=True)
            os.makedirs(baseline_dir, exist_ok=True)
            print(f"\n  [{done}/{total_combos}]  seed={seed}  α={alpha_val}")

            set_seeds(seed)

            # ── 1. Train BPR ──────────────────────────────────
            print(f"    Training BPR (seed={seed}, epochs={EPOCHS}) ...")
            trainer = BPRTrainer(
                embedding_dim=EMBEDDING_DIM,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                seed=seed,
                device=DEVICE,
            )
            trainer.fit(train_df)

            # ── 2. Remap fixed long-tail to internal indices ──
            # Items absent = zero training interactions = structural limit
            (long_tail_int,
             short_head_int,
             item_pop_int,
             n_tail_missing) = remap_fixed_long_tail(
                tail_ids_fixed, head_ids_fixed,
                item_pop_original, trainer.item_map
            )
            print(f"    Long-tail (fixed): {len(long_tail_int):,} present  "
                  f"{n_tail_missing:,} missing from training")

            # ── 3. Remap dataframes ───────────────────────────
            test_df_int  = remap_df(test_df,  trainer.user_map, trainer.item_map)
            train_df_int = remap_df(train_df, trainer.user_map, trainer.item_map)

            # ── 4. Ground truth (vectorised) ──────────────────
            ground_truth  = build_ground_truth(test_df_int)

            # ── 5. User profiles (vectorised) ─────────────────
            user_profiles = build_user_profiles(train_df_int)

            # ── 6. Generate candidates ────────────────────────
            print(f"    Generating top-{RERANK_FROM} candidates ...")
            test_users = test_df["user_id"].unique().tolist()
            candidates = generate_candidate_pool(
                trainer, test_users, RERANK_FROM
            )

            # ── 7. Candidate pool analysis ────────────────────
            pool_analysis = analyse_candidate_pool(
                candidates, long_tail_int, RERANK_FROM
            )
            print(f"    Pool LT fraction : "
                  f"{pool_analysis['mean_lt_fraction_in_pool']:.3f}  "
                  f"(theoretical max APLT: "
                  f"{pool_analysis['theoretical_max_aplt']:.3f})")
            print(f"    Users zero LT    : "
                  f"{pool_analysis['users_with_zero_lt_in_pool']:,} "
                  f"({pool_analysis['pct_users_zero_lt']:.1f}%)")

            # ── 8. Evaluate baseline ──────────────────────────
            print(f"    Evaluating baseline ...")
            baseline_eval = evaluate_baseline(
                candidates, ground_truth,
                item_pop_int, long_tail_int, k=K
            )
            agg = baseline_eval['aggregate']
            print(f"    NDCG@{K}: {agg[f'NDCG@{K}']:.4f}  "
                  f"APLT@{K}: {agg[f'APLT@{K}']:.4f}  "
                  f"ARP@{K}: {agg[f'ARP@{K}']:.1f}")

            # ── 9. Save everything ────────────────────────────
            with open(os.path.join(save_dir, "candidates.pkl"), "wb") as f:
                pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(save_dir, "user_profiles.pkl"), "wb") as f:
                pickle.dump(user_profiles, f, protocol=pickle.HIGHEST_PROTOCOL)

            eval_bundle = {
                "ground_truth":    dict(ground_truth),
                "item_popularity": item_pop_int,
                "long_tail_items": list(long_tail_int),
                "short_head_items":list(short_head_int),
                "long_tail_source":"fixed_from_alpha_1.0",
                "n_tail_missing":  n_tail_missing,
                "pool_analysis":   pool_analysis,
            }
            with open(os.path.join(save_dir, "eval_bundle.pkl"), "wb") as f:
                pickle.dump(eval_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

            meta = {
                "seed":            seed,
                "alpha":           alpha_val,
                "n_users":         len(candidates),
                "n_items":         len(trainer.item_map),
                "n_candidates":    RERANK_FROM,
                "n_long_tail":     len(long_tail_int),
                "n_short_head":    len(short_head_int),
                "n_tail_missing":  n_tail_missing,
                "n_test_users":    len(ground_truth),
                "epochs":          EPOCHS,
                "embedding_dim":   EMBEDDING_DIM,
                "long_tail_fixed": True,
                **{f"pool_{k}": v for k, v in pool_analysis.items()},
            }
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            # Baseline metrics
            baseline_data = {
                "seed":    seed,
                "alpha":   alpha_val,
                "metrics": agg,
                "n_evaluated": len(baseline_eval['per_user_ndcg']),
                "n_tail_missing": n_tail_missing,
                **{f"pool_{k}": v for k, v in pool_analysis.items()},
            }
            with open(os.path.join(baseline_dir,
                                   "baseline_metrics.json"), "w") as f:
                json.dump(baseline_data, f, indent=2)

            with open(os.path.join(baseline_dir,
                                   "per_user_ndcg.json"), "w") as f:
                json.dump({str(k): v for k, v in
                           baseline_eval['per_user_ndcg'].items()}, f)

            with open(os.path.join(baseline_dir,
                                   "per_user_aplt.json"), "w") as f:
                json.dump({str(k): v for k, v in
                           baseline_eval['per_user_aplt'].items()}, f)

            all_baseline_results.append(baseline_data)
            print(f"    ✓ Saved → {save_dir}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"STAGE 1 COMPLETE — {done}/{total_combos} runs")
    print(f"{'=' * 65}")

    if all_baseline_results:
        summary_df = pd.DataFrame([
            {
                'alpha': r['alpha'],
                'seed':  r['seed'],
                **r['metrics'],
                'n_tail_missing': r.get('n_tail_missing', 0),
                'pool_mean_lt':   r.get('pool_mean_lt_fraction_in_pool', 0),
            }
            for r in all_baseline_results
        ])
        summary_df.to_csv(
            os.path.join(BASELINE_METRICS_DIR, "baseline_summary.csv"),
            index=False
        )

        print(f"\n  Baseline summary (mean across {len(SEEDS)} seeds):")
        print(f"  {'α':>6} {'NDCG@10':>10} {'APLT@10':>10} "
              f"{'ARP@10':>10} {'TailMissing':>12} {'PoolLT':>8}")
        print(f"  {'-'*60}")

        for alpha_val in ALPHA_LEVELS:
            rows = summary_df[summary_df['alpha'] == alpha_val]
            if rows.empty:
                continue
            print(f"  {alpha_val:>6.2f} "
                  f"{rows[f'NDCG@{K}'].mean():>10.4f} "
                  f"{rows[f'APLT@{K}'].mean():>10.4f} "
                  f"{rows[f'ARP@{K}'].mean():>10.1f} "
                  f"{rows['n_tail_missing'].mean():>12.0f} "
                  f"{rows['pool_mean_lt'].mean():>8.3f}")

    print(f"\n  Candidates → {CANDIDATES_DIR}")
    print(f"  Metrics    → {BASELINE_METRICS_DIR}")
    print(f"  Run stage2 next.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_stage1()