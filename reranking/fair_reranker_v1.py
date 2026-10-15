"""
FA*IR Fair Reranker — corrected threshold direction
====================================================
Zehlike et al. "FA*IR: A Fair Top-k Ranking Algorithm"

BUG FIXED: Original code found the smallest m where P(X ≥ m) ≤ α
(upper tail), which required ~9 protected items out of 10 at τ=0.5.
This made FA*IR act as a "greedy max long-tail" reranker.

CORRECT: Find the LARGEST m where P(X < m) ≤ α (lower tail).
P(X < m) = P(X ≤ m-1) = binom.cdf(m-1, i, p)

This gives the minimum required protected count per position —
small enough to be achievable but large enough to ensure fairness.
"""

from scipy.stats import binom
import numpy as np


class FAIRReranker:
    def __init__(self, p=0.3, alpha=0.05, k=10):
        """
        Args:
            p:     Target minimum proportion of long-tail items (τ)
            alpha: Significance level for the binomial test
            k:     Recommendation list length
        """
        self.p     = p
        self.alpha = alpha
        self.k     = k
        self.m     = self._compute_thresholds()

    def _compute_thresholds(self):
        """
        Compute m_i: minimum protected items required at each position i.

        m_i = largest m such that P(X < m) ≤ α
             where X ~ Binomial(i, p)

        P(X < m) = P(X ≤ m-1) = binom.cdf(m-1, i, p)

        Interpretation: if fewer than m_i protected items appear by
        position i, that is statistically unlikely under proportion p
        (probability ≤ α), so we enforce the minimum.

        Example: p=0.5, α=0.05, i=10
          m=1: P(X<1) = 0.5^10 ≈ 0.001 ≤ 0.05 → found_m = 1
          m=2: P(X<2) = binom.cdf(1,10,0.5) ≈ 0.011 ≤ 0.05 → found_m = 2
          m=3: P(X<3) = binom.cdf(2,10,0.5) ≈ 0.055 > 0.05 → stop
          Result: threshold_10 = 2  (require ≥ 2 protected in top-10)

        Old (buggy) result: threshold_10 = 9  (required ≥ 9 protected)
        """
        thresholds = []
        for i in range(1, self.k + 1):
            found_m = 0  # default: no minimum required at this position
            for m in range(1, i + 1):
                # P(X < m) = P(X ≤ m-1)
                prob = binom.cdf(m - 1, i, self.p)
                if prob <= self.alpha:
                    found_m = m  # this m still satisfies the constraint
                else:
                    break  # P(X < m) > α — stop, use last valid found_m
            thresholds.append(found_m)
        return thresholds

    def rerank(self, candidates, scores, protected_mask):
        """
        Rerank to satisfy FA*IR fairness constraints.

        Args:
            candidates:     List of item IDs (top-RERANK_FROM from BPR)
            scores:         Dict {item_id: relevance_score}
            protected_mask: Dict {item_id: True if long-tail}

        Returns:
            List of reranked item IDs (length = min(k, len(candidates)))
        """
        protected   = sorted(
            [(i, scores[i]) for i in candidates if protected_mask.get(i, False)],
            key=lambda x: x[1], reverse=True
        )
        unprotected = sorted(
            [(i, scores[i]) for i in candidates if not protected_mask.get(i, False)],
            key=lambda x: x[1], reverse=True
        )

        result          = []
        p_idx = u_idx   = 0
        protected_count = 0

        for pos in range(self.k):
            required = self.m[pos]  # min protected needed up to this position

            if protected_count < required:
                # Must take a protected item to meet the threshold
                if p_idx < len(protected):
                    result.append(protected[p_idx][0])
                    p_idx += 1
                    protected_count += 1
                elif u_idx < len(unprotected):
                    # No protected left — fall back to unprotected
                    result.append(unprotected[u_idx][0])
                    u_idx += 1
                else:
                    break
            else:
                # Threshold met — pick highest relevance item
                has_p = p_idx < len(protected)
                has_u = u_idx < len(unprotected)

                if has_p and has_u:
                    if protected[p_idx][1] >= unprotected[u_idx][1]:
                        result.append(protected[p_idx][0])
                        p_idx += 1
                        protected_count += 1
                    else:
                        result.append(unprotected[u_idx][0])
                        u_idx += 1
                elif has_p:
                    result.append(protected[p_idx][0])
                    p_idx += 1
                    protected_count += 1
                elif has_u:
                    result.append(unprotected[u_idx][0])
                    u_idx += 1
                else:
                    break

        return result

    def get_thresholds(self):
        return self.m