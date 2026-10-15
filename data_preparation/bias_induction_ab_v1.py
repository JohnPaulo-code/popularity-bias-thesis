
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  — final agreed settings
# ─────────────────────────────────────────────────────────────
INPUT_FILE       = '../original_data/ratings_Books.csv'
OUTPUT_DIR       = '../data_preparation/biased_datasets/AmazonBooks_v1'
TARGET_SIZE      = 750_000
ALPHA_LEVELS     = [1.0, 1.2, 1.5, 1.75, 2.0]   # 5 levels
RANDOM_SEED      = 42
HEAD_MASS        = 0.80
MIN_INTERACTIONS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


def load_and_filter_amazon(path: str):
    print("  Reading file ...")
    try:
        df = pd.read_csv(
            path,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            header=None
        )
    except Exception:
        df = pd.read_csv(
            path,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            header=None, sep='\t'
        )

    print(f"  Raw interactions: {len(df):,}")

    # Iterative 20-core — repeat until stable
    prev_len, iteration = -1, 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        u_counts = df['user_id'].value_counts()
        i_counts = df['item_id'].value_counts()
        mask = (
            df['user_id'].isin(u_counts[u_counts >= MIN_INTERACTIONS].index) &
            df['item_id'].isin(i_counts[i_counts >= MIN_INTERACTIONS].index)
        )
        df = df[mask].copy()
        print(f"    Pass {iteration}: {len(df):,} interactions")

    print(f"  Stable after {iteration} passes: {len(df):,} interactions")

    # GLOBAL mappings — fixed across ALL alpha levels
    user_map = {u: i for i, u in enumerate(sorted(df['user_id'].unique()))}
    item_map = {it: i for i, it in enumerate(sorted(df['item_id'].unique()))}
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)

    return df, user_map, item_map


def compute_fixed_long_tail(df, item_map, head_mass=HEAD_MASS):
    """
    Computed ONCE from full original filtered data.
    Same 16,018 tail items tracked at every alpha level.
    """
    counts = df['item_id'].value_counts()
    sorted_counts = counts.sort_values(ascending=False)
    cum_mass = sorted_counts.cumsum() / sorted_counts.sum()

    head_ids = set(sorted_counts[cum_mass <= head_mass].index)
    if not head_ids:
        head_ids = {sorted_counts.index[0]}
    tail_ids = set(sorted_counts.index) - head_ids

    head_idx = {item_map[i] for i in head_ids if i in item_map}
    tail_idx  = {item_map[i] for i in tail_ids  if i in item_map}

    print(f"\n  Fixed long-tail definition (original filtered data):")
    print(f"    Total items : {len(counts):,}")
    print(f"    Head items  : {len(head_ids):,} "
          f"({len(head_ids)/len(counts)*100:.1f}%)")
    print(f"    Tail items  : {len(tail_ids):,} "
          f"({len(tail_ids)/len(counts)*100:.1f}%)")

    return head_ids, tail_ids, head_idx, tail_idx, dict(counts)


def induce_bias(df: pd.DataFrame, alpha: float,
                target_n: int, rng) -> pd.DataFrame:

    pop_norm = df['item_id'].value_counts(normalize=True)
    weights  = (df['item_id'].map(pop_norm)
                              .fillna(1e-10)
                              .values) ** (alpha - 1.0)
    weights  = weights / weights.sum()
    idx = rng.choice(len(df), size=target_n, replace=True, p=weights)
    return (df.iloc[idx]
             [['user_id', 'item_id', 'user_idx', 'item_idx']]
             .copy().reset_index(drop=True))


def per_user_split(df, test_ratio=0.20,
                   min_interactions=5, seed=RANDOM_SEED):
    """
    Per-user split guarantees every user has test interactions.
    Users with fewer than min_interactions go entirely to train.
    """
    train_rows, test_rows = [], []
    for user, group in df.groupby('user_id'):
        if len(group) < min_interactions:
            train_rows.append(group)
            continue
        n_test    = max(1, int(len(group) * test_ratio))
        test_idx  = group.sample(n=n_test, random_state=seed).index
        train_idx = group.index.difference(test_idx)
        train_rows.append(group.loc[train_idx])
        test_rows.append(group.loc[test_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return ((2 * np.dot(np.arange(1, n + 1), x))
            / (n * x.sum())) - (n + 1) / n


def compute_bias_metrics(biased_df, head_ids_fixed,
                          tail_ids_fixed, item_pop_original):
    """All metrics use fixed long-tail definition for comparability."""
    bc    = biased_df['item_id'].value_counts()
    total = len(biased_df)

    g           = gini(bc.values)
    top1_share  = float(bc.iloc[0] / total)
    top10_share = float(bc.iloc[:10].sum() / total)

    head_ints      = biased_df[biased_df['item_id'].isin(head_ids_fixed)].shape[0]
    head_ratio     = head_ints / total
    tail_ratio     = (total - head_ints) / total

    # KEY METRIC: how many genuine tail items survive in this dataset
    # Decreasing tail_coverage with alpha = candidate pool depletion
    tail_items_present = len(set(bc.index) & tail_ids_fixed)
    tail_coverage      = tail_items_present / len(tail_ids_fixed)

    # ARP uses original data popularity — consistent across alpha levels
    arp = float(biased_df['item_id'].map(item_pop_original).fillna(0).mean())

    return {
        'gini':               round(g, 4),
        'top1_share':         round(top1_share, 4),
        'top10_share':        round(top10_share, 4),
        'head_ratio':         round(head_ratio, 4),
        'tail_ratio':         round(tail_ratio, 4),
        'tail_items_present': tail_items_present,
        'tail_items_missing': len(tail_ids_fixed) - tail_items_present,
        'tail_coverage':      round(tail_coverage, 4),
        'arp':                round(arp, 1),
        'n_users':            biased_df['user_id'].nunique(),
        'n_items':            len(bc),
        'n_interactions':     total,
    }


def save_sparse(df, n_users, n_items, path):
    mat = csr_matrix(
        (np.ones(len(df)), (df['user_idx'], df['item_idx'])),
        shape=(n_users, n_items)
    )
    scipy.sparse.save_npz(path, mat)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
print("Loading and filtering Amazon Books ...")
original_df, user_map, item_map = load_and_filter_amazon(INPUT_FILE)
n_users, n_items = len(user_map), len(item_map)
print(f"  Users: {n_users:,}  Items: {n_items:,}  "
      f"Interactions: {len(original_df):,}")

# Save original filtered
orig_dir = os.path.join(OUTPUT_DIR, 'original_filtered')
os.makedirs(orig_dir, exist_ok=True)
original_df.to_csv(os.path.join(orig_dir, 'original_filtered.csv'),
                   index=False)

# ── Fixed long-tail (computed ONCE, never changes) ────────────
(head_ids_fixed, tail_ids_fixed,
 head_idx_fixed, tail_idx_fixed,
 item_pop_original) = compute_fixed_long_tail(original_df, item_map)

# Save everything stage1 needs
fixed_lt = {
    'head_item_ids': list(head_ids_fixed),
    'tail_item_ids': list(tail_ids_fixed),
    'head_item_idx': list(head_idx_fixed),
    'tail_item_idx': list(tail_idx_fixed),
    'n_head':        len(head_ids_fixed),
    'n_tail':        len(tail_ids_fixed),
    'n_total_items': n_items,
    'head_mass':     HEAD_MASS,
    'source':        'original_filtered_data',
}
with open(os.path.join(OUTPUT_DIR, 'fixed_long_tail.json'), 'w') as f:
    json.dump(fixed_lt, f, indent=2)

# Save original item popularity for ARP computation in stage1
item_pop_serialisable = {str(k): int(v)
                          for k, v in item_pop_original.items()}
with open(os.path.join(OUTPUT_DIR, 'item_pop_original.json'), 'w') as f:
    json.dump(item_pop_serialisable, f)

# Save global mappings for stage1 index remapping
with open(os.path.join(OUTPUT_DIR, 'item_map.json'), 'w') as f:
    json.dump(item_map, f)
with open(os.path.join(OUTPUT_DIR, 'user_map.json'), 'w') as f:
    json.dump(user_map, f)

print(f"\n  Saved: fixed_long_tail.json, item_pop_original.json, "
      f"item_map.json, user_map.json")

# ── Bias induction ────────────────────────────────────────────
print(f"\nInducing bias — {len(ALPHA_LEVELS)} levels: {ALPHA_LEVELS}")
print(f"Target size: {TARGET_SIZE:,}  (replace=True)")
print("=" * 65)

summary_rows = []

for alpha in ALPHA_LEVELS:
    print(f"\n  α = {alpha}")
    alpha_dir = os.path.join(OUTPUT_DIR, f"alpha_{alpha}")
    os.makedirs(alpha_dir, exist_ok=True)

    biased_df = induce_bias(original_df, alpha, TARGET_SIZE, rng)
    m = compute_bias_metrics(biased_df, head_ids_fixed,
                              tail_ids_fixed, item_pop_original)

    print(f"    Gini              : {m['gini']:.4f}")
    print(f"    Top-10 share      : {m['top10_share']:.4f}")
    print(f"    Head ratio        : {m['head_ratio']:.4f} (fixed)")
    print(f"    Tail coverage     : {m['tail_coverage']*100:.1f}% "
          f"({m['tail_items_present']:,}/{len(tail_ids_fixed):,} items)")
    print(f"    Tail items missing: {m['tail_items_missing']:,} "
          f"← cannot be promoted by post-processing")
    print(f"    ARP               : {m['arp']:.1f}")

    train_df, test_df = per_user_split(biased_df, test_ratio=0.20,
                                        seed=RANDOM_SEED)
    print(f"    Train / Test      : {len(train_df):,} / {len(test_df):,}")
    print(f"    Test users        : {test_df['user_id'].nunique():,}")

    train_df.to_csv(os.path.join(alpha_dir, 'train.csv'), index=False)
    test_df.to_csv( os.path.join(alpha_dir, 'test.csv'),  index=False)
    save_sparse(train_df, n_users, n_items,
                os.path.join(alpha_dir, 'train.npz'))

    stats = {'alpha': alpha, **m,
             'n_train': len(train_df), 'n_test': len(test_df),
             'n_test_users': int(test_df['user_id'].nunique())}
    with open(os.path.join(alpha_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    summary_rows.append(stats)
    print(f"    Saved → {alpha_dir}")

# ── Monotonicity verification ─────────────────────────────────
print("\n" + "=" * 65)
print("MONOTONICITY VERIFICATION")
print("=" * 65)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'bias_summary.csv'),
                  index=False)

checks = {
    'Gini (↑)':          summary_df['gini'].values,
    'Head ratio (↑)':    summary_df['head_ratio'].values,
    'ARP (↑)':           summary_df['arp'].values,
    'Top-10 share (↑)':  summary_df['top10_share'].values,
    'Tail coverage (↓)': summary_df['tail_coverage'].values,
}

# With 5 levels: idx 0=α1.0, 1=α1.2, 2=α1.5, 3=α1.75, 4=α2.0
print(f"\n  {'Metric':<24} {'α=1.0':>8} {'α=1.5':>8} "
      f"{'α=2.0':>8}   Mono?   ρ")
print(f"  {'-'*65}")

all_ok = True
for name, vals in checks.items():
    up = '↑' in name
    mono = all(
        vals[i] <= vals[i+1] if up else vals[i] >= vals[i+1]
        for i in range(len(vals)-1)
    )
    if not mono:
        all_ok = False

    rho, pval = spearmanr(ALPHA_LEVELS, vals)
    status = '✓' if mono else '✗'
    print(f"  {name:<24} "
          f"{vals[0]:>8.4f} {vals[2]:>8.4f} {vals[-1]:>8.4f}"
          f"   {status}    ρ={rho:+.3f} p={pval:.4f}")

print()
if all_ok:
    print("  ✓ ALL METRICS MONOTONIC — bias induction valid.")
else:
    print("  Check non-monotonic metrics above.")
    print("  ρ > 0.95 with minor adjacency issues is acceptable.")

print(f"\n  Summary → {os.path.join(OUTPUT_DIR, 'bias_summary.csv')}")
print("=" * 65)