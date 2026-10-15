"""
Cross-Model Robustness Experiment
===================================
Mirrors BPR stage1/stage2 structure exactly.

STAGE 1 — Train model, generate candidates, save to disk (HEAVY)
  Output: candidates.pkl, user_profiles.pkl, eval_bundle.pkl,
          meta.json, baseline_metrics.json
  Resume: skipped if all 5 files exist

STAGE 2 — Load saved candidates, run mitigation (LIGHT)
  Output: mitigation_results.json, per_user_*.json
  Resume: skipped if mitigation_results.json exists

To fix a mitigation bug without retraining:
  → Delete mitigation_results.json files only (see reset_mitigation())
  → Rerun script — stage1 is skipped, only stage2 reruns

Structure:
  5 alpha × 3 seeds × 2 models × 2 datasets = 60 training runs
"""

import os
import sys
import json
import pickle
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_training.lightgcn_trainer import LightGCNTrainer
from model_training.neumf_trainer    import NeuMFTrainer
from reranking.xquad_binary      import BinaryxQuAD
from reranking.xquad_smooth      import SmoothxQuAD
from reranking.fair_reranker_v1  import FAIRReranker

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

DATASETS = {
    'MovieLens': {
        'data_dir': os.path.join(BASE, r"data_preparation\biased_datasets\MovieLens_v1"),
        'out_dir':  os.path.join(BASE, r"cross_model\{model}\MovieLens"),
    },
    'AmazonBooks': {
        'data_dir': os.path.join(BASE, r"data_preparation\biased_datasets\AmazonBooks_v1"),
        'out_dir':  os.path.join(BASE, r"cross_model\{model}\AmazonBooks"),
    },
}

MODELS = {
    'LightGCN': {
        'cls':    LightGCNTrainer,
        'kwargs': {'embedding_dim': 64, 'n_layers': 3, 'epochs': 50,
                   'lr': 0.02, 'batch_size': 2048, 'device': 'cuda'},
    },
    'NeuMF': {
        'cls':    NeuMFTrainer,
        'kwargs': {'embedding_dim': 64, 'mlp_dims': [64, 32, 16, 8],
                   'dropout': 0.2, 'epochs': 50,
                   'lr': 0.02, 'batch_size': 2048, 'device': 'cuda'},
    },
}

ACTIVE_MODELS   = ['LightGCN', 'NeuMF']
ACTIVE_DATASETS = ['MovieLens', 'AmazonBooks']

SEEDS        = [42, 123, 456]
ALPHA_LEVELS = [1.0, 1.2, 1.5, 1.75, 2.0]
K            = 10
RERANK_FROM  = 100

# Nash consensus parameters — fixed, no sweep
FIXED_PARAMS = {
    'BinaryxQuAD': 0.4,
    'SmoothxQuAD': 0.9,
    'FAIR':        0.9,
}

# Chunk size for candidate generation — prevents GPU OOM
# 256 users × 38,121 items × 4 bytes ≈ 39 MB per chunk
PRED_CHUNK = 256

# ── Stage1 files (saved by training run) ──────────────────────
STAGE1_FILES = [
    'candidates.pkl',
    'user_profiles.pkl',
    'eval_bundle.pkl',
    'meta.json',
    'baseline_metrics.json',
]

# ── Stage2 files (saved by mitigation run) ────────────────────
STAGE2_FILES = [
    'mitigation_results.json',
]


# ─────────────────────────────────────────────────────────────
# RESUME CHECKS — independent for stage1 and stage2
# ─────────────────────────────────────────────────────────────
def is_stage1_complete(save_dir: str) -> bool:
    """True if all training/candidate files exist."""
    return all(os.path.exists(os.path.join(save_dir, f))
               for f in STAGE1_FILES)


def is_stage2_complete(save_dir: str) -> bool:
    """True if mitigation results exist."""
    return os.path.exists(
        os.path.join(save_dir, 'mitigation_results.json')
    )


# ─────────────────────────────────────────────────────────────
# FIXED LONG-TAIL LOADING
# ─────────────────────────────────────────────────────────────
def load_fixed_long_tail(data_dir: str):
    """
    Load fixed long-tail definition computed once from the original
    filtered distribution during bias induction.
    Returns original item IDs (not internal indices).
    Remapping to trainer indices happens after training,
    once the trainer key type is known.
    """
    with open(os.path.join(data_dir, 'fixed_long_tail.json')) as f:
        fixed = json.load(f)
    tail_ids = set(fixed['tail_item_ids'])
    head_ids = set(fixed['head_item_ids'])
    print(f"  Fixed LT: {len(head_ids):,} head / {len(tail_ids):,} tail")
    return tail_ids, head_ids


def load_original_item_popularity(data_dir: str):
    """
    Load item popularity from original filtered data (ARP reference).

    item_pop_original.json stores all keys as strings.
    MovieLens trainer maps use integer keys — int() conversion required.
    Amazon Books ASINs stay as strings (int() fails and is caught).
    """
    with open(os.path.join(data_dir, 'item_pop_original.json')) as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        try:
            result[int(k)] = int(v)       # MovieLens: '42' → 42
        except (ValueError, TypeError):
            result[k] = int(v)            # Amazon Books: ASIN kept as str
    return result


def remap_fixed_long_tail(tail_ids, head_ids, item_pop_original,
                           trainer_item_map):
    """
    Remap original item IDs to trainer's internal integer indices.

    Detects trainer key type (int for MovieLens, str for Amazon Books)
    and converts item IDs accordingly. Items absent from trainer_item_map
    had zero training interactions — structural depletion, not a bug.
    """
    sample_key = next(iter(trainer_item_map))
    use_str    = isinstance(sample_key, str)

    def to_key(x):
        return str(x) if use_str else x

    tail_int = {trainer_item_map[to_key(iid)]
                for iid in tail_ids if to_key(iid) in trainer_item_map}
    head_int = {trainer_item_map[to_key(iid)]
                for iid in head_ids if to_key(iid) in trainer_item_map}
    pop_int  = {trainer_item_map[to_key(k)]: cnt
                for k, cnt in item_pop_original.items()
                if to_key(k) in trainer_item_map}

    n_missing = len(tail_ids) - len(tail_int)
    return tail_int, head_int, pop_int, n_missing


# ─────────────────────────────────────────────────────────────
# CANDIDATE GENERATION — chunked to avoid OOM
# ─────────────────────────────────────────────────────────────
def generate_candidate_pool(trainer, test_users,
                             pool_size=RERANK_FROM,
                             chunk_size=PRED_CHUNK) -> dict:
    """
    Generate top-pool_size candidates per user, chunked by user
    to prevent GPU OOM on large datasets (Amazon Books: 35K×38K items).
    Uses numpy argpartition O(n) instead of full sort O(n log n).
    """
    valid_users = [u for u in test_users if u in trainer.user_map]
    if not valid_users:
        return {}

    all_item_t = torch.arange(len(trainer.item_map)).to(trainer.device)
    n_items    = len(trainer.item_map)
    all_scores = []

    for i in range(0, len(valid_users), chunk_size):
        chunk_u = valid_users[i: i + chunk_size]
        u_idx   = torch.LongTensor(
            [trainer.user_map[u] for u in chunk_u]
        ).to(trainer.device)
        chunk_s = trainer._batch_predict(u_idx, all_item_t)
        all_scores.append(chunk_s)

    scores_np = np.vstack(all_scores)

    candidates = {}
    for row_idx, user_id in enumerate(valid_users):
        user_scores = scores_np[row_idx].copy()
        pos_set = trainer.user_pos_items.get(
            trainer.user_map[user_id], set()
        )
        if pos_set:
            user_scores[
                np.array(list(pos_set), dtype=np.int64)
            ] = -np.inf

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
# EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(recs: dict, ground_truth: dict,
             item_popularity: dict, long_tail_items: set,
             k: int = K) -> tuple:
    """Vectorised evaluation — no EvaluationRunner dependency."""
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
        recall_val = (sum(1 for item in predicted if item in true_items)
                      / len(true_items))
        aplt_val   = sum(1 for item in predicted
                         if item in long_tail_items) / len(predicted)

        for item in predicted:
            total_pop    += item_popularity.get(item, 0)
            n_recs_total += 1

        ndcg_scores.append(ndcg_val)
        recall_scores.append(recall_val)
        aplt_scores.append(aplt_val)
        per_user_ndcg[user] = ndcg_val
        per_user_aplt[user] = aplt_val

    arp = total_pop / n_recs_total if n_recs_total > 0 else 0.0
    agg = {
        f'NDCG@{k}':   float(np.mean(ndcg_scores))  if ndcg_scores  else 0.0,
        f'Recall@{k}': float(np.mean(recall_scores)) if recall_scores else 0.0,
        f'APLT@{k}':   float(np.mean(aplt_scores))  if aplt_scores  else 0.0,
        f'ARP@{k}':    float(arp),
    }
    return agg, per_user_ndcg, per_user_aplt


# ─────────────────────────────────────────────────────────────
# RERANKING — fixed Nash-consensus parameters
# ─────────────────────────────────────────────────────────────
def run_xquad(reranker_cls, user_data, long_tail, short_head, lam):
    reranker = reranker_cls(
        lambda_param=lam, rerank_from=RERANK_FROM, rerank_to=K
    )
    recs = {}
    for user, ud in user_data.items():
        recs[user] = reranker.rerank(
            user_idx        = user,
            candidate_items = ud['items'],
            base_scores     = ud['norm_scores'],
            user_profile    = ud['profile'],
            long_tail_items = long_tail,
            short_head_items= short_head,
        )
    return recs


def run_fair(user_data, long_tail, tau):
    reranker = FAIRReranker(p=tau, alpha=0.05, k=K)
    recs     = {}
    for user, ud in user_data.items():
        pm       = {item: (item in long_tail) for item in ud['items']}
        reranked = reranker.rerank(ud['items'], ud['scores'], pm)
        recs[user] = [(item, ud['scores'].get(item, 0.0))
                      for item in reranked]
    return recs


def build_user_data(candidates, user_profiles):
    """Precompute per-user dict for rerankers from saved candidates."""
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


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(v) for v in obj]
    return obj


def build_ground_truth(test_df_int):
    return test_df_int.groupby('user_idx')['item_idx'].apply(set).to_dict()


def build_user_profiles_df(train_df_int):
    return train_df_int.groupby('user_idx')['item_idx'].apply(set).to_dict()


def remap_df(df, user_map, item_map):
    df = df.copy()
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    df.dropna(subset=['user_idx', 'item_idx'], inplace=True)
    df['user_idx'] = df['user_idx'].astype(int)
    df['item_idx'] = df['item_idx'].astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# STAGE 1 — Train + Generate Candidates (HEAVY)
# ─────────────────────────────────────────────────────────────
def run_stage1(model_name, trainer_cls, trainer_kwargs,
               dataset_name, data_dir,
               alpha_val, seed,
               tail_ids_fixed, head_ids_fixed, item_pop_original,
               save_dir):
    """
    Train model, generate top-100 candidates, evaluate baseline,
    save all stage1 files. Does NOT run mitigation.

    Saved files:
      candidates.pkl      — {user_idx: [(item_idx, score), ...]}
      user_profiles.pkl   — {user_idx: set(item_idx)}
      eval_bundle.pkl     — ground_truth, item_pop, long_tail, short_head
      meta.json           — run metadata
      baseline_metrics.json — NDCG, APLT, ARP, Recall at K
    """
    set_seeds(seed)
    os.makedirs(save_dir, exist_ok=True)

    # Load biased data for this alpha
    alpha_dir = os.path.join(data_dir, f"alpha_{alpha_val}")
    train_df  = pd.read_csv(os.path.join(alpha_dir, "train.csv"))
    test_df   = pd.read_csv(os.path.join(alpha_dir, "test.csv"))

    # ── Train model ───────────────────────────────────────────
    print(f"    [S1] Training {model_name} "
          f"(seed={seed}, epochs={trainer_kwargs['epochs']}) ...")
    trainer = trainer_cls(
        seed=seed,
        **{k: v for k, v in trainer_kwargs.items() if k != 'seed'}
    )
    trainer.fit(train_df)

    # ── Remap fixed long-tail to internal indices ──────────────
    (long_tail_int, short_head_int,
     item_pop_int, n_tail_missing) = remap_fixed_long_tail(
        tail_ids_fixed, head_ids_fixed,
        item_pop_original, trainer.item_map
    )
    print(f"    [S1] Long-tail: {len(long_tail_int):,} present  "
          f"{n_tail_missing:,} missing from training")

    # ── Remap dataframes ──────────────────────────────────────
    test_df_int  = remap_df(test_df,  trainer.user_map, trainer.item_map)
    train_df_int = remap_df(train_df, trainer.user_map, trainer.item_map)
    ground_truth  = build_ground_truth(test_df_int)
    user_profiles = build_user_profiles_df(train_df_int)

    # ── Generate candidates ───────────────────────────────────
    print(f"    [S1] Generating top-{RERANK_FROM} candidates ...")
    test_users = test_df['user_id'].unique().tolist()
    candidates = generate_candidate_pool(
        trainer, test_users, RERANK_FROM, PRED_CHUNK
    )

    # ── Evaluate baseline ─────────────────────────────────────
    user_data = build_user_data(candidates, user_profiles)
    baseline_recs = {
        u: [(item, ud['scores'].get(item, 0.0))
            for item in ud['items'][:K]]
        for u, ud in user_data.items()
    }
    bl_agg, bl_ndcg, bl_aplt = evaluate(
        baseline_recs, ground_truth, item_pop_int, long_tail_int
    )
    print(f"    [S1] Baseline  "
          f"NDCG@{K}={bl_agg[f'NDCG@{K}']:.4f}  "
          f"APLT@{K}={bl_agg[f'APLT@{K}']:.4f}  "
          f"ARP@{K}={bl_agg[f'ARP@{K}']:.1f}")

    # ── Save stage1 files ─────────────────────────────────────
    with open(os.path.join(save_dir, 'candidates.pkl'), 'wb') as f:
        pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dir, 'user_profiles.pkl'), 'wb') as f:
        pickle.dump(user_profiles, f, protocol=pickle.HIGHEST_PROTOCOL)

    eval_bundle = {
        'ground_truth':     dict(ground_truth),
        'item_popularity':  item_pop_int,
        'long_tail_items':  list(long_tail_int),
        'short_head_items': list(short_head_int),
        'long_tail_source': 'fixed_from_alpha_1.0',
        'n_tail_missing':   n_tail_missing,
    }
    with open(os.path.join(save_dir, 'eval_bundle.pkl'), 'wb') as f:
        pickle.dump(eval_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = ensure_serializable({
        'model':          model_name,
        'dataset':        dataset_name,
        'seed':           seed,
        'alpha':          alpha_val,
        'n_users':        len(candidates),
        'n_items':        len(trainer.item_map),
        'n_candidates':   RERANK_FROM,
        'n_long_tail':    len(long_tail_int),
        'n_tail_missing': n_tail_missing,
        'n_test_users':   len(ground_truth),
        'epochs':         trainer_kwargs['epochs'],
        'embedding_dim':  trainer_kwargs['embedding_dim'],
    })
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    baseline_data = ensure_serializable({
        'model':          model_name,
        'dataset':        dataset_name,
        'seed':           seed,
        'alpha':          alpha_val,
        'metrics':        bl_agg,
        'n_evaluated':    len(bl_ndcg),
        'n_tail_missing': n_tail_missing,
        'per_user_ndcg':  {str(u): v for u, v in bl_ndcg.items()},
        'per_user_aplt':  {str(u): v for u, v in bl_aplt.items()},
    })
    with open(os.path.join(save_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(baseline_data, f, indent=2)

    print(f"    [S1] ✓ Stage1 saved → {save_dir}")


# ─────────────────────────────────────────────────────────────
# STAGE 2 — Load Candidates + Mitigate (LIGHT)
# ─────────────────────────────────────────────────────────────
def run_stage2(model_name, dataset_name, alpha_val, seed, save_dir):
    """
    Load saved stage1 candidates and eval_bundle, apply fixed-parameter
    mitigation, save results. No model training — can be rerun in
    minutes if mitigation has a bug.

    To rerun mitigation only:
      delete mitigation_results.json → rerun script → stage1 skipped

    Saved files:
      mitigation_results.json — aggregate metrics per method
      per_user_{method}.json  — per-user NDCG and APLT
    """
    # ── Load saved stage1 outputs ─────────────────────────────
    with open(os.path.join(save_dir, 'candidates.pkl'), 'rb') as f:
        candidates = pickle.load(f)

    with open(os.path.join(save_dir, 'user_profiles.pkl'), 'rb') as f:
        user_profiles = pickle.load(f)

    with open(os.path.join(save_dir, 'eval_bundle.pkl'), 'rb') as f:
        bundle = pickle.load(f)

    ground_truth   = bundle['ground_truth']
    item_pop_int   = bundle['item_popularity']
    long_tail_int  = set(bundle['long_tail_items'])
    short_head_int = set(bundle['short_head_items'])

    print(f"    [S2] Loaded {len(candidates):,} users  "
          f"LT={len(long_tail_int):,}")

    # ── Build user data for rerankers ─────────────────────────
    user_data = build_user_data(candidates, user_profiles)

    # ── Apply each fixed-parameter method ─────────────────────
    mit_results = {}
    for method_name, lam in FIXED_PARAMS.items():
        if method_name in ('BinaryxQuAD', 'SmoothxQuAD'):
            cls  = (BinaryxQuAD if method_name == 'BinaryxQuAD'
                    else SmoothxQuAD)
            recs = run_xquad(cls, user_data,
                             long_tail_int, short_head_int, lam)
        else:
            recs = run_fair(user_data, long_tail_int, lam)

        agg, pu_ndcg, pu_aplt = evaluate(
            recs, ground_truth, item_pop_int, long_tail_int
        )
        mit_results[method_name] = {
            'param':        lam,
            'metrics':      agg,
            'per_user_ndcg': {str(u): v for u, v in pu_ndcg.items()},
            'per_user_aplt': {str(u): v for u, v in pu_aplt.items()},
        }
        print(f"    [S2] {method_name}(λ={lam})  "
              f"NDCG@{K}={agg[f'NDCG@{K}']:.4f}  "
              f"APLT@{K}={agg[f'APLT@{K}']:.4f}  "
              f"ARP@{K}={agg[f'ARP@{K}']:.1f}")

    # ── Save stage2 files ─────────────────────────────────────
    mit_out = ensure_serializable({
        'model':        model_name,
        'dataset':      dataset_name,
        'seed':         seed,
        'alpha':        alpha_val,
        'fixed_params': FIXED_PARAMS,
        'methods': {
            mname: {
                'param':   mdata['param'],
                'metrics': mdata['metrics'],
            }
            for mname, mdata in mit_results.items()
        },
    })
    with open(os.path.join(save_dir, 'mitigation_results.json'), 'w') as f:
        json.dump(mit_out, f, indent=2)

    for mname, mdata in mit_results.items():
        with open(os.path.join(save_dir,
                               f'per_user_{mname}.json'), 'w') as f:
            json.dump({'ndcg': mdata['per_user_ndcg'],
                       'aplt': mdata['per_user_aplt']}, f)

    print(f"    [S2] ✓ Stage2 saved → {save_dir}")


# ─────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────
def aggregate_results(model_name, dataset_name, out_root):
    """Collect all stage1 + stage2 results into a single CSV."""
    rows = []
    for seed in SEEDS:
        for alpha in ALPHA_LEVELS:
            save_dir = os.path.join(out_root,
                                     f"seed_{seed}", f"alpha_{alpha}")
            bl_path  = os.path.join(save_dir, 'baseline_metrics.json')
            mit_path = os.path.join(save_dir, 'mitigation_results.json')

            if not os.path.exists(bl_path):
                continue

            with open(bl_path) as f:
                bl = json.load(f)

            rows.append({
                'model': model_name, 'dataset': dataset_name,
                'seed': seed, 'alpha': alpha,
                'method': 'Baseline', 'param': None,
                **bl['metrics'],
            })

            if os.path.exists(mit_path):
                with open(mit_path) as f:
                    mit = json.load(f)
                for mname, mdata in mit['methods'].items():
                    rows.append({
                        'model': model_name, 'dataset': dataset_name,
                        'seed': seed, 'alpha': alpha,
                        'method': mname, 'param': mdata['param'],
                        **mdata['metrics'],
                    })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(out_root, 'all_results_combined.csv'), index=False
    )
    summary = df.groupby(['model', 'dataset', 'method', 'alpha']).agg(
        NDCG_mean  =('NDCG@10',   'mean'), NDCG_std  =('NDCG@10',   'std'),
        APLT_mean  =('APLT@10',   'mean'), APLT_std  =('APLT@10',   'std'),
        ARP_mean   =('ARP@10',    'mean'), ARP_std   =('ARP@10',    'std'),
        Recall_mean=('Recall@10', 'mean'), Recall_std=('Recall@10', 'std'),
    ).round(4)
    summary.to_csv(os.path.join(out_root, 'summary_table.csv'))
    return df


# ─────────────────────────────────────────────────────────────
# UTILITY — Reset mitigation only (no retraining needed)
# ─────────────────────────────────────────────────────────────
def reset_mitigation(models=None, datasets=None):
    """
    Delete only mitigation_results.json and per_user_*.json files.
    Stage1 files (candidates, eval_bundle, etc.) are preserved.
    After calling this, rerun the script — stage1 is skipped,
    only stage2 reruns.

    Usage:
        reset_mitigation()                          # all
        reset_mitigation(models=['LightGCN'])       # one model
        reset_mitigation(datasets=['MovieLens'])    # one dataset
    """
    models   = models   or ACTIVE_MODELS
    datasets = datasets or ACTIVE_DATASETS
    deleted  = 0

    for model_name in models:
        for dataset_name in datasets:
            out_root = DATASETS[dataset_name]['out_dir'].format(
                model=model_name
            )
            for seed in SEEDS:
                for alpha in ALPHA_LEVELS:
                    save_dir = os.path.join(out_root,
                                             f"seed_{seed}",
                                             f"alpha_{alpha}")
                    # Delete mitigation_results.json
                    mit = os.path.join(save_dir, 'mitigation_results.json')
                    if os.path.exists(mit):
                        os.remove(mit)
                        deleted += 1

                    # Delete per_user_*.json files
                    if os.path.isdir(save_dir):
                        for fname in os.listdir(save_dir):
                            if fname.startswith('per_user_'):
                                os.remove(os.path.join(save_dir, fname))

    print(f"  Deleted mitigation_results.json for {deleted} runs.")
    print(f"  Stage1 files untouched. Rerun script to redo mitigation.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_cross_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total     = (len(ACTIVE_MODELS) * len(ACTIVE_DATASETS)
                 * len(ALPHA_LEVELS) * len(SEEDS))
    done      = 0

    print(f"\n{'=' * 65}")
    print(f"  CROSS-MODEL ROBUSTNESS EXPERIMENT")
    print(f"  Timestamp   : {timestamp}")
    print(f"  Models      : {ACTIVE_MODELS}")
    print(f"  Datasets    : {ACTIVE_DATASETS}")
    print(f"  Alphas      : {ALPHA_LEVELS}")
    print(f"  Seeds       : {SEEDS}")
    print(f"  Total runs  : {total}")
    print(f"  Fixed λ/τ   : {FIXED_PARAMS}")
    print(f"  Stage1 files: {STAGE1_FILES}")
    print(f"  Stage2 files: {STAGE2_FILES}")
    print(f"{'=' * 65}")
    print(f"  To redo mitigation only (no retraining):")
    print(f"    reset_mitigation()  then rerun")
    print(f"{'=' * 65}")

    for model_name in ACTIVE_MODELS:
        if model_name not in MODELS:
            print(f"\n  SKIP {model_name}: not in MODELS config")
            continue

        mcfg        = MODELS[model_name]
        trainer_cls = mcfg['cls']
        trainer_kw  = dict(mcfg['kwargs'])

        for dataset_name in ACTIVE_DATASETS:
            if dataset_name not in DATASETS:
                continue

            dcfg     = DATASETS[dataset_name]
            data_dir = dcfg['data_dir']
            out_root = dcfg['out_dir'].format(model=model_name)
            os.makedirs(out_root, exist_ok=True)

            print(f"\n{'─' * 65}")
            print(f"  {model_name} × {dataset_name}")

            # Load fixed definitions ONCE per (model, dataset)
            tail_ids, head_ids = load_fixed_long_tail(data_dir)
            item_pop_orig = load_original_item_popularity(data_dir)

            for alpha_val in ALPHA_LEVELS:
                print(f"\n  α={alpha_val}")

                for seed in SEEDS:
                    done    += 1
                    save_dir = os.path.join(out_root,
                                             f"seed_{seed}",
                                             f"alpha_{alpha_val}")
                    tag = (f"[{done}/{total}] "
                           f"{model_name}/{dataset_name} "
                           f"seed={seed} α={alpha_val}")

                    # ── Stage 1 ───────────────────────────────
                    if is_stage1_complete(save_dir):
                        print(f"  {tag}  [S1 SKIP — candidates exist]")
                    else:
                        print(f"\n  {tag}")
                        try:
                            run_stage1(
                                model_name, trainer_cls, trainer_kw,
                                dataset_name, data_dir,
                                alpha_val, seed,
                                tail_ids, head_ids, item_pop_orig,
                                save_dir,
                            )
                        except Exception as e:
                            print(f"    [S1] ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            continue   # skip stage2 if stage1 failed

                    # ── Stage 2 ───────────────────────────────
                    if is_stage2_complete(save_dir):
                        print(f"  {tag}  [S2 SKIP — mitigation exists]")
                    else:
                        try:
                            run_stage2(
                                model_name, dataset_name,
                                alpha_val, seed, save_dir,
                            )
                        except Exception as e:
                            print(f"    [S2] ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

            # Aggregate after all seeds × alphas complete
            print(f"\n  Aggregating {model_name}/{dataset_name} ...")
            df = aggregate_results(model_name, dataset_name, out_root)
            if df is not None:
                print(f"\n  Summary ({model_name}/{dataset_name}):")
                print(f"  {'Method':>14} {'α=1.0 NDCG':>12} "
                      f"{'α=1.0 APLT':>12} {'α=1.0 ARP':>11} "
                      f"{'α=2.0 APLT':>12}")
                print(f"  {'-' * 62}")
                for method in ['Baseline', 'BinaryxQuAD',
                               'SmoothxQuAD', 'FAIR']:
                    m = df[df['method'] == method]
                    if m.empty:
                        continue
                    n10 = m[m['alpha']==1.0]['NDCG@10'].mean()
                    a10 = m[m['alpha']==1.0]['APLT@10'].mean()
                    r10 = m[m['alpha']==1.0]['ARP@10'].mean()
                    a20 = m[m['alpha']==2.0]['APLT@10'].mean()
                    print(f"  {method:>14} {n10:>12.4f} "
                          f"{a10:>12.4f} {r10:>11.1f} {a20:>12.4f}")

    print(f"\n{'=' * 65}")
    print(f"  CROSS-MODEL EXPERIMENT COMPLETE")
    print(f"  Results → cross_model/{{model}}/{{dataset}}/")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_cross_model()
