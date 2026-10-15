
import numpy as np
from collections import defaultdict
from scipy import stats
from reranking.base_reranker import xQuADReranker

class BinaryxQuAD(xQuADReranker):
    """
    Binary xQuAD: Promotes at least one item from uncovered category.
    """

    def rerank(self, user_idx, candidate_items, base_scores,
               user_profile, long_tail_items, short_head_items):
        """Re-rank using Binary xQuAD."""
        # Normalize base scores
        max_score = max(base_scores.values()) if base_scores else 1.0
        if max_score > 0:
            normalized_scores = {k: v / max_score for k, v in base_scores.items()}
        else:
            normalized_scores = base_scores

        # Compute user category preferences P(d|u)
        category_prefs = self.compute_user_category_preference(user_profile, long_tail_items)

        # Initialize result list S
        S = []
        remaining_items = set(candidate_items[:self.rerank_from])

        # Track which categories have been covered
        covered = {'long_tail': False, 'short_head': False}

        # Iteratively select items
        for _ in range(min(self.rerank_to, len(remaining_items))):
            best_item = None
            best_score = -np.inf

            for item in remaining_items:
                # Compute P(v|u) - base score
                p_v_u = normalized_scores.get(item, 0.0)

                # Compute P(v, S'|u) - diversity bonus
                diversity_bonus = 0.0

                # Check item category
                if item in long_tail_items:
                    item_category = 'long_tail'
                else:
                    item_category = 'short_head'

                # P(v|d) = 1 if item belongs to category d
                p_v_d = 1.0

                # Product term: equals 1 if category not covered yet
                if not covered[item_category]:
                    prod_term = 1.0
                else:
                    prod_term = 0.0

                diversity_bonus = category_prefs[item_category] * p_v_d * prod_term

                # Final score
                score = (1 - self.lambda_param) * p_v_u + self.lambda_param * diversity_bonus

                if score > best_score:
                    best_score = score
                    best_item = item

            # Add best item to result
            if best_item is not None:
                S.append((best_item, base_scores.get(best_item, 0.0)))
                remaining_items.remove(best_item)

                # Mark category as covered
                if best_item in long_tail_items:
                    covered['long_tail'] = True
                else:
                    covered['short_head'] = True

        return S
