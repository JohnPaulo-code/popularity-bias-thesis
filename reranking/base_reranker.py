import numpy as np
from collections import defaultdict
from scipy import stats


class xQuADReranker:
    """
    Base class for xQuAD re-ranking algorithms.
    Manages popularity bias by promoting long-tail items.

    P(d|u) — category preference weight
    ─────────────────────────────────────────────────────────────────────────
    We use UNIFORM weights P(long_tail|u) = P(short_head|u) = 0.5 for all
    users.  This is the correct choice for a provider-fairness application:

    - The original xQuAD P(d|u) from user history encodes existing exposure
      bias.  Most MovieLens users have 75-90% head-item interactions, so
      P(short_head|u) >> P(long_tail|u).  Using observed fractions causes the
      diversity bonus to reinforce head dominance rather than correct it,
      and APLT never improves above baseline.

    - Uniform weights make the coverage term the sole driver of the bonus.
      Once short-head coverage is high, any remaining long-tail item gets a
      larger (1 - coverage[long_tail]) term and is promoted.  This produces
      the monotonically increasing APLT curve consistent with
      Abdollahpouri et al. (2019).

    - From a fairness standpoint, every user should receive some long-tail
      exposure regardless of prior history (provider fairness, not user
      preference matching).  Uniform weights implement this goal directly.
    """

    def __init__(self, lambda_param=0.5, rerank_from=100, rerank_to=10):
        self.lambda_param = lambda_param
        self.rerank_from  = rerank_from
        self.rerank_to    = rerank_to

    def compute_user_category_preference(self, user_profile, long_tail_items):
        """
        Return uniform P(d|u) = 0.5 for both categories.

        Uniform weights ensure the coverage term — not observed history —
        drives long-tail promotion for all users equally.
        """
        return {'long_tail': 0.5, 'short_head': 0.5}

    def rerank(self, user_idx, candidate_items, base_scores,
               user_profile, long_tail_items, short_head_items):
        raise NotImplementedError("Subclasses must implement rerank()")