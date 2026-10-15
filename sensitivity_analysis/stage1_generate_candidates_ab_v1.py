# stage1_generate_candidates_ab.py
import os
import json
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from model_training.bpr_trainer_v1 import BPRTrainer
from evaluation.evaluation_v1 import RecommenderMetrics

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_DIR             = r"D:\Final_thesis\thesis_project\data_preparation\biased_datasets\AmazonBooks_v1"
CANDIDATES_DIR       = r"D:\Final_thesis\thesis_project\sensitivity_analysis\candidates_ab_v1"
BASELINE_METRICS_DIR = r"D:\Final_thesis\thesis_project\sensitivity_analysis\baseline_metrics_ab_v1"

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
SEEDS         = [42, 123, 456]
ALPHA_LEVELS  = [1.0, 1.2, 1.5, 1.75, 2.0]
RERANK_FROM   = 100
EPOCHS        = 50
EMBEDDING_DIM = 64
LR            = 0.02
BATCH_SIZE    = 2048
DEVICE        = "cuda"
K             = 10


# ─────────────────────────────────────────────────────────────
# FIXED LONG-TAIL LOADING
# FIX: load as-is from JSON — no type conversion here.
# Type matching happens in remap_fixed_long_tail where we have
# access to trainer.item_map to detect the correct key type.
#
# Amazon Books has mixed item IDs:
#   - Non-numeric ASINs: 'B00CW6BA2C' (string)
#   - Numeric ISBNs:     '0316052094' (string with leading zero)
# Converting numeric strings to int loses the leading zero and
# breaks matching against trainer.item_map (which uses strings
# because pandas reads mixed columns as object dtype).
# ─────────────────────────────────────────────────────────────
def load_fixed_long_tail(data_dir: str):
    path = os.path.join(data_dir, 'fixed_long_tail.json')
    with open(path) as f:
        fixed = json.load(f)

    # Load exactly as stored in JSON — no conversion
    tail_ids = set(fixed['tail_item_ids'])
    head_ids = set(fixed['head_item_ids'])

    print(f"  Fixed long-tail loaded: "
          f"{len(head_ids):,} head / {len(tail_ids):,} tail items")
    print(f"  ID type sample: {type(next(iter(tail_ids))).__name__} "
          f"— e.g. {next(iter(tail_ids))}")
    return tail_ids, head_ids


def load_original_item_popularity(data_dir: str):
    """
    Load item popularity from original filtered data.
    FIX: keep string keys as-is — type matching happens
    in remap_fixed_long_tail after the trainer is available.
    """
    path = os.path.join(data_dir, 'item_pop_original.json')
    with open(path) as f:
        raw = json.load(f)
    # Keys are strings in JSON (saved with str() in bias induction)
    # Values are interaction counts
    return {k: int(v) for k, v in raw.items()}


def remap_fixed_long_tail(tail_ids, head_ids, item_pop_original,
                           trainer_item_map):
    """
    Remap fixed original IDs to trainer's internal indices.

    FIX: dynamically detect trainer key type and convert IDs
    to match, instead of blindly converting to int.

    MovieLens:     trainer.item_map has int/int64 keys
                   → keep JSON ints as-is

    Amazon Books:  trainer.item_map has str keys (pandas reads
                   mixed ASIN/ISBN column as object dtype)
                   → convert all IDs to str, preserving leading
                   zeros in numeric ISBNs like '0316052094'
    """
    # Detect key type from one sample trainer key
    sample_key = next(iter(trainer_item_map))
    use_str    = isinstance(sample_key, str)

    def to_key(x):
        """Convert x to match trainer_item_map key type."""
        return str(x) if use_str else x

    tail_int = {trainer_item_map[to_key(iid)]
                for iid in tail_ids
                if to_key(iid) in trainer_item_map}

    head_int = {trainer_item_map[to_key(iid)]
                for iid in head_ids
                if to_key(iid) in trainer_item_map}

    pop_int  = {trainer_item_map[to_key(k)]: cnt
                for k, cnt in item_pop_original.items()
                if to_key(k) in trainer_item_map}

    n_missing = len(tail_ids) - len(tail_int)
    return tail_int, head_int, pop_int, n_missing


# ─────────────────────────────────────────────────────────────
# CANDIDATE POOL ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyse_candidate_pool(candidates: dict,
                            long_tail_int: set,
                            pool_size: int = 100) -> dict:
    per_user_lt = []
    users_zero  = 0
    for user, cand_list in candidates.items():
        lt_count = sum(1 for item, _ in cand_list[:pool_size]
                       if item in long_tail_int)
        per_user_lt.append(lt_count / pool_size)
        if lt_count == 0:
            users_zero += 1
    return {
        'mean_lt_fraction_in_pool':   float(np.mean(per_user_lt)),
        'median_lt_fraction_in_pool': float(np.median(per_user_lt)),
        'users_with_zero_lt_in_pool': users_zero,
        'pct_users_zero_lt':          users_zero / len(candidates) * 100,
        'theoretical_max_aplt':       float(np.mean(
            [min(1.0, f) for f in per_user_lt]
        )),
    }


# ─────────────────────────────────────────────────────────────
# CANDIDATE GENERATION — numpy optimised
# ─────────────────────────────────────────────────────────────
def generate_candidate_pool(trainer, test_users,
                             pool_size: int = 100) -> dict:
    valid_users  = [u for u in test_users if u in trainer.user_map]
    user_indices = torch.LongTensor(
        [trainer.user_map[u] for u in valid_users]
    ).to(trainer.device)
    all_item_t = torch.arange(len(trainer.item_map)).to(trainer.device)

    scores_np = trainer._batch_predict(user_indices, all_item_t)

    candidates = {}
    n_items    = len(trainer.item_map)

    for row_idx, user_id in enumerate(valid_users):
        user_scores = scores_np[row_idx].copy()
        pos_set = trainer.user_pos_items.get(
            trainer.user_map[user_id], set()
        )
        if pos_set:
            user_scores[np.array(list(pos_set), dtype=np.int64)] = -np.inf

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
    ndcg_scores, recall_scores = [], []
    per_user_ndcg, per_user_aplt = {}, {}

    for user_id, recs in candidates.items():
        true_items = ground_truth.get(user_id, set())
        if not true_items:
            continue
        predicted = [item for item, _ in recs[:k]]
        ndcg      = RecommenderMetrics.ndcg_at_k(true_items, predicted, k)
        recall    = RecommenderMetrics.recall_at_k(true_items, predicted, k)
        n_lt      = sum(1 for item in predicted if item in long_tail_items)
        aplt      = n_lt / len(predicted) if predicted else 0.0
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        per_user_ndcg[user_id] = ndcg
        per_user_aplt[user_id] = aplt

    total_pop, n_recs = 0.0, 0
    for user_id, recs in candidates.items():
        for item, _ in recs[:k]:
            total_pop += item_popularity.get(item, 0)
            n_recs    += 1
    arp = total_pop / n_recs if n_recs > 0 else 0.0

    return {
        'aggregate': {
            f'NDCG@{k}':   float(np.mean(ndcg_scores))   if ndcg_scores   else 0.0,
            f'Recall@{k}': float(np.mean(recall_scores))  if recall_scores else 0.0,
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
    """Vectorised — replaces slow iterrows()."""
    return (
        test_df_int.groupby("user_idx")["item_idx"]
        .apply(set).to_dict()
    )


def build_user_profiles(train_df_int: pd.DataFrame) -> dict:
    """Vectorised — replaces slow iterrows()."""
    return (
        train_df_int.groupby("user_idx")["item_idx"]
        .apply(set).to_dict()
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
    os.makedirs(CANDIDATES_DIR,       exist_ok=True)
    os.makedirs(BASELINE_METRICS_DIR, exist_ok=True)

    # ── Load fixed definitions ONCE — as-is from JSON ─────────
    tail_ids_fixed, head_ids_fixed = load_fixed_long_tail(DATA_DIR)
    item_pop_original = load_original_item_popularity(DATA_DIR)

    total_combos        = len(ALPHA_LEVELS) * len(SEEDS)
    done                = 0
    all_baseline_results = []

    print(f"\n{'=' * 65}")
    print(f"  STAGE 1 — Amazon Books BPR + Candidate Generation")
    print(f"  Alpha levels : {ALPHA_LEVELS}")
    print(f"  Seeds        : {SEEDS}")
    print(f"  Total runs   : {total_combos}")
    print(f"  Fixed tail   : {len(tail_ids_fixed):,} items")
    print(f"  Fixed head   : {len(head_ids_fixed):,} items")
    print(f"{'=' * 65}")

    # ── OUTER LOOP: alpha levels (load each CSV once) ─────────
    for alpha_val in ALPHA_LEVELS:
        alpha_dir = os.path.join(DATA_DIR, f"alpha_{alpha_val}")
        train_csv = os.path.join(alpha_dir, "train.csv")
        test_csv  = os.path.join(alpha_dir, "test.csv")

        if not os.path.exists(train_csv):
            print(f"\n  ERROR: Missing train.csv in {alpha_dir}")
            continue

        print(f"\n{'─' * 65}")
        print(f"  α = {alpha_val}  — loading dataset")

        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        print(f"    Train: {len(train_df):,}  "
              f"Test: {len(test_df):,}  "
              f"Users: {test_df['user_id'].nunique():,}")

        # ── INNER LOOP: seeds ─────────────────────────────────
        for seed in SEEDS:
            done     += 1
            save_dir  = os.path.join(CANDIDATES_DIR,
                                      f"seed_{seed}",
                                      f"alpha_{alpha_val}")
            base_dir  = os.path.join(BASELINE_METRICS_DIR,
                                      f"seed_{seed}",
                                      f"alpha_{alpha_val}")

            # ── Resume: skip if already complete ─────────────
            required = ["candidates.pkl", "user_profiles.pkl",
                        "eval_bundle.pkl", "meta.json"]
            if (all(os.path.exists(os.path.join(save_dir, f))
                    for f in required) and
                    os.path.exists(
                        os.path.join(base_dir, "baseline_metrics.json")
                    )):
                print(f"  [{done}/{total_combos}] SKIP  "
                      f"seed={seed}  α={alpha_val}")
                try:
                    with open(os.path.join(
                            base_dir, "baseline_metrics.json")) as f:
                        bd = json.load(f)
                        bd['seed']  = seed
                        bd['alpha'] = alpha_val
                        all_baseline_results.append(bd)
                except Exception:
                    pass
                continue

            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(base_dir, exist_ok=True)
            print(f"\n  [{done}/{total_combos}]  "
                  f"seed={seed}  α={alpha_val}")

            set_seeds(seed)

            # ── 1. Train BPR ──────────────────────────────────
            print(f"    Training BPR "
                  f"(seed={seed}, epochs={EPOCHS}) ...")
            trainer = BPRTrainer(
                embedding_dim=EMBEDDING_DIM,
                epochs=EPOCHS,
                lr=LR,
                batch_size=BATCH_SIZE,
                seed=seed,
                device=DEVICE,
            )
            trainer.fit(train_df)

            # ── 2. Remap fixed long-tail ──────────────────────
            # remap_fixed_long_tail detects trainer key type
            # (str for Amazon Books) and converts IDs to match.
            # This fixes the leading-zero loss bug where
            # '0316052094' was wrongly converted to int 316052094.
            (long_tail_int,
             short_head_int,
             item_pop_int,
             n_tail_missing) = remap_fixed_long_tail(
                tail_ids_fixed, head_ids_fixed,
                item_pop_original, trainer.item_map
            )
            print(f"    Trainer key type : "
                  f"{type(next(iter(trainer.item_map))).__name__}")
            print(f"    Long-tail (fixed): "
                  f"{len(long_tail_int):,} present  "
                  f"{n_tail_missing:,} missing from training")

            # ── 3. Remap dataframes ───────────────────────────
            test_df_int  = remap_df(test_df,
                                     trainer.user_map, trainer.item_map)
            train_df_int = remap_df(train_df,
                                     trainer.user_map, trainer.item_map)

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
                  f"(max APLT: "
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
                  f"ARP@{K}:  {agg[f'ARP@{K}']:.1f}")

            # ── 9. Save everything ────────────────────────────
            with open(os.path.join(save_dir,
                                   "candidates.pkl"), "wb") as f:
                pickle.dump(candidates, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(save_dir,
                                   "user_profiles.pkl"), "wb") as f:
                pickle.dump(user_profiles, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            eval_bundle = {
                "ground_truth":     dict(ground_truth),
                "item_popularity":  item_pop_int,
                "long_tail_items":  list(long_tail_int),
                "short_head_items": list(short_head_int),
                "long_tail_source": "fixed_from_alpha_1.0",
                "n_tail_missing":   n_tail_missing,
                "pool_analysis":    pool_analysis,
            }
            with open(os.path.join(save_dir,
                                   "eval_bundle.pkl"), "wb") as f:
                pickle.dump(eval_bundle, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            meta = {
                "seed":             seed,
                "alpha":            alpha_val,
                "dataset":          "AmazonBooks",
                "n_users":          len(candidates),
                "n_items":          len(trainer.item_map),
                "n_candidates":     RERANK_FROM,
                "n_long_tail":      len(long_tail_int),
                "n_short_head":     len(short_head_int),
                "n_tail_missing":   n_tail_missing,
                "n_test_users":     len(ground_truth),
                "epochs":           EPOCHS,
                "embedding_dim":    EMBEDDING_DIM,
                "long_tail_fixed":  True,
                "trainer_key_type": type(
                    next(iter(trainer.item_map))).__name__,
                **{f"pool_{k}": v
                   for k, v in pool_analysis.items()},
            }
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            baseline_data = {
                "seed":           seed,
                "alpha":          alpha_val,
                "metrics":        agg,
                "n_evaluated":    len(baseline_eval['per_user_ndcg']),
                "n_tail_missing": n_tail_missing,
                **{f"pool_{k}": v
                   for k, v in pool_analysis.items()},
            }
            with open(os.path.join(base_dir,
                                   "baseline_metrics.json"), "w") as f:
                json.dump(baseline_data, f, indent=2)

            with open(os.path.join(base_dir,
                                   "per_user_ndcg.json"), "w") as f:
                json.dump({str(k): v for k, v in
                           baseline_eval['per_user_ndcg'].items()}, f)

            with open(os.path.join(base_dir,
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
        summary_rows = []
        for r in all_baseline_results:
            summary_rows.append({
                'alpha': r['alpha'],
                'seed':  r['seed'],
                **r['metrics'],
                'n_tail_missing': r.get('n_tail_missing', 0),
                'pool_mean_lt':   r.get(
                    'pool_mean_lt_fraction_in_pool', 0),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(BASELINE_METRICS_DIR, "baseline_summary.csv"),
            index=False
        )

        print(f"\n  Baseline summary (mean across {len(SEEDS)} seeds):")
        print(f"  {'α':>6} {'NDCG@10':>10} {'APLT@10':>10} "
              f"{'ARP@10':>10} {'TailMissing':>12} {'PoolLT':>8}")
        print(f"  {'-' * 60}")

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