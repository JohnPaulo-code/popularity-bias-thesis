import numpy as np
from collections import defaultdict
from scipy import stats
from reranking.base_reranker import xQuADReranker


class SmoothxQuAD(xQuADReranker):
    """
    Smooth xQuAD: Uses ratio of items covering each category.

    At each step, the diversity bonus for a candidate item is:
        bonus = P(d|u) * P(v|d) * (1 - coverage_ratio[d])

    where coverage_ratio[d] = |{items in S belonging to category d}| / |S|.

    FIX (v2): After placing any item, BOTH category coverage ratios are
    recomputed from the full current list S.  Previously only the placed
    item's category was updated, leaving the other ratio frozen at a stale
    value as |S| grew — causing incorrect diversity bonuses.
    """

    def rerank(self, user_idx, candidate_items, base_scores,
               user_profile, long_tail_items, short_head_items):
        """Re-rank using Smooth xQuAD."""

        # ── Normalise base scores ─────────────────────────────────────────────
        max_score = max(base_scores.values()) if base_scores else 1.0
        if max_score > 0:
            normalized_scores = {k: v / max_score for k, v in base_scores.items()}
        else:
            normalized_scores = base_scores

        # ── User category preferences P(d|u) ─────────────────────────────────
        category_prefs = self.compute_user_category_preference(
            user_profile, long_tail_items
        )

        # ── Initialise ────────────────────────────────────────────────────────
        S = []                                          # selected items so far
        remaining_items = set(candidate_items[:self.rerank_from])

        # Coverage ratio: fraction of items in S that belong to each category.
        # Both start at 0.0 (S is empty).
        coverage = {'long_tail': 0.0, 'short_head': 0.0}

        # ── Greedy selection ──────────────────────────────────────────────────
        for _ in range(min(self.rerank_to, len(remaining_items))):
            best_item  = None
            best_score = -np.inf

            for item in remaining_items:
                p_v_u = normalized_scores.get(item, 0.0)

                # Category of the candidate item
                item_category = (
                    'long_tail' if item in long_tail_items else 'short_head'
                )

                # P(v|d) = 1 (item fully represents its own category)
                p_v_d = 1.0

                # Diversity bonus
                prod_term     = 1.0 - coverage[item_category]
                diversity_bonus = (
                    category_prefs[item_category] * p_v_d * prod_term
                )

                score = (
                    (1 - self.lambda_param) * p_v_u
                    + self.lambda_param * diversity_bonus
                )

                if score > best_score:
                    best_score = score
                    best_item  = item

            if best_item is None:
                break

            S.append((best_item, base_scores.get(best_item, 0.0)))
            remaining_items.remove(best_item)

            # ── FIX: recompute BOTH coverage ratios after every placement ─────
            # len(S) has grown, so both ratios must be refreshed even if only
            # one category's count changed.
            n_lt = sum(1 for it, _ in S if it in long_tail_items)
            n_s  = len(S)
            coverage['long_tail']  = n_lt / n_s
            coverage['short_head'] = (n_s - n_lt) / n_s

        return S