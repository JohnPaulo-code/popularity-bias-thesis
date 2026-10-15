

import numpy as np
from collections import defaultdict


class RecommenderMetrics:
    """Collection of evaluation metrics for recommender systems."""

    @staticmethod
    def dcg_at_k(relevances, k):
        relevances = np.array(relevances)[:k]
        if relevances.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, relevances.size + 2))
        return np.sum(relevances / discounts)

    @staticmethod
    def ndcg_at_k(true_items, predicted_items, k=10):
        relevances = [1 if item in true_items else 0
                      for item in predicted_items[:k]]
        dcg  = RecommenderMetrics.dcg_at_k(relevances, k)
        ideal = [1] * min(len(true_items), k) + \
                [0] * max(0, k - len(true_items))
        idcg = RecommenderMetrics.dcg_at_k(ideal, k)
        return 0.0 if idcg == 0 else dcg / idcg

    @staticmethod
    def precision_at_k(true_items, predicted_items, k=10):
        if k == 0:
            return 0.0
        predicted_k = set(predicted_items[:k])
        return len(predicted_k & set(true_items)) / k

    @staticmethod
    def recall_at_k(true_items, predicted_items, k=10):
        if len(true_items) == 0:
            return 0.0
        predicted_k = set(predicted_items[:k])
        return len(predicted_k & set(true_items)) / len(true_items)

    @staticmethod
    def average_recommendation_popularity(recommendations_dict,
                                           item_popularity, k=10):
        """ARP — average popularity of recommended items."""
        total, n_users = 0.0, 0
        for user, recs in recommendations_dict.items():
            user_pop, count = 0.0, 0
            for item, _ in recs[:k]:
                user_pop += item_popularity.get(item, 0)
                count += 1
            if count > 0:
                total += user_pop / count
                n_users += 1
        return total / n_users if n_users > 0 else 0.0

    @staticmethod
    def average_percentage_long_tail(recommendations_dict,
                                      long_tail_items, k=10):
        total, n_users = 0.0, 0
        for user, recs in recommendations_dict.items():
            n_lt, count = 0, 0
            for item, _ in recs[:k]:
                if item in long_tail_items:
                    n_lt += 1
                count += 1
            if count > 0:
                total += n_lt / count
                n_users += 1
        return total / n_users if n_users > 0 else 0.0

    @staticmethod
    def average_coverage_long_tail(recommendations_dict,
                                    long_tail_items, k=10):
        """ACLT — fraction of tail items that appear in any recommendation."""
        if len(long_tail_items) == 0:
            return 0.0
        recommended_lt = set()
        for user, recs in recommendations_dict.items():
            for item, _ in recs[:k]:
                if item in long_tail_items:
                    recommended_lt.add(item)
        return len(recommended_lt) / len(long_tail_items)

    @staticmethod
    def aggregate_diversity(recommendations_dict, k=10):
        """Total unique items recommended across all users."""
        all_items = set()
        for user, recs in recommendations_dict.items():
            for item, _ in recs[:k]:
                all_items.add(item)
        return len(all_items)


class EvaluationRunner:
    """Helper class to run full evaluation on test data."""

    def __init__(self, test_data, train_data, item_popularity,
                 long_tail_items, short_head_items):
        self.item_popularity  = item_popularity
        self.long_tail_items  = long_tail_items
        self.short_head_items = short_head_items

        # Vectorised ground truth — replaces iterrows() which is
        # extremely slow on DataFrames with hundreds of thousands of rows
        self.ground_truth = (
            test_data.groupby('user_idx')['item_idx']
            .apply(set)
            .to_dict()
        )

    def evaluate_recommendations(self, recommendations_dict, k=10):
        """
        Evaluate a set of recommendations.

        Args:
            recommendations_dict: {user_idx: [(item_idx, score), ...]}
            k: Cutoff position

        Returns:
            Dict of metric_name -> value
        """
        ndcg_scores, precision_scores, recall_scores = [], [], []

        for user, recs in recommendations_dict.items():
            true_items = self.ground_truth.get(user, set())
            if not true_items:
                continue
            predicted = [item for item, _ in recs]
            ndcg_scores.append(
                RecommenderMetrics.ndcg_at_k(true_items, predicted, k)
            )
            precision_scores.append(
                RecommenderMetrics.precision_at_k(true_items, predicted, k)
            )
            recall_scores.append(
                RecommenderMetrics.recall_at_k(true_items, predicted, k)
            )

        metrics = {
            f'NDCG@{k}':      np.mean(ndcg_scores)      if ndcg_scores      else 0.0,
            f'Precision@{k}': np.mean(precision_scores)  if precision_scores else 0.0,
            f'Recall@{k}':    np.mean(recall_scores)     if recall_scores    else 0.0,
            f'ARP@{k}': RecommenderMetrics.average_recommendation_popularity(
                recommendations_dict, self.item_popularity, k),
            f'APLT@{k}': RecommenderMetrics.average_percentage_long_tail(
                recommendations_dict, self.long_tail_items, k),
            f'ACLT@{k}': RecommenderMetrics.average_coverage_long_tail(
                recommendations_dict, self.long_tail_items, k),
            f'AggregateDiversity@{k}': RecommenderMetrics.aggregate_diversity(
                recommendations_dict, k),
        }
        return metrics

    def evaluate_per_user(self, recommendations_dict: dict,
                           k: int = 10) -> dict:
        """Per-user NDCG for statistical testing."""
        per_user_ndcg = {}
        for user_id, rec_list in recommendations_dict.items():
            true_items = self.ground_truth.get(user_id, set())
            if not true_items:
                per_user_ndcg[user_id] = 0.0
                continue
            rec_items = [item_id for item_id, _ in rec_list[:k]]
            dcg = sum(
                1.0 / np.log2(i + 2)
                for i, item in enumerate(rec_items)
                if item in true_items
            )
            ideal_count = min(len(true_items), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
            per_user_ndcg[user_id] = dcg / idcg if idcg > 0 else 0.0
        return per_user_ndcg

    def evaluate_per_user_aplt(self, recommendations_dict: dict,
                                k: int = 10) -> dict:
        """Per-user APLT for statistical testing."""
        per_user_aplt = {}
        for user_id, rec_list in recommendations_dict.items():
            rec_items = [item_id for item_id, _ in rec_list[:k]]
            if not rec_items:
                per_user_aplt[user_id] = 0.0
                continue
            n_lt = sum(1 for item in rec_items
                       if item in self.long_tail_items)
            per_user_aplt[user_id] = n_lt / len(rec_items)
        return per_user_aplt


class PerUserMetrics:
    def __init__(self):
        self.metrics = defaultdict(dict)

    def add_user_result(self, user_id, algorithm, lambda_val, metrics):
        key = f"{algorithm}_λ{lambda_val}"
        self.metrics[user_id][key] = metrics

    def get_paired_values(self, metric_name, condition_a, condition_b):
        paired_a, paired_b = [], []
        for user_id in self.metrics:
            if (condition_a in self.metrics[user_id] and
                    condition_b in self.metrics[user_id]):
                paired_a.append(self.metrics[user_id][condition_a][metric_name])
                paired_b.append(self.metrics[user_id][condition_b][metric_name])
        return np.array(paired_a), np.array(paired_b)