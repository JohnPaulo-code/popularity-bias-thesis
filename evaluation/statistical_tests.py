import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ── Default fallback parameters (used only if Nash file not found) ────────────
DEFAULT_LAMBDAS = {
    'BinaryxQuAD': 0.5,
    'SmoothxQuAD': 0.5,
    'FAIR': 0.3,
}


def load_nash_lambdas(nash_json_path: str) -> dict:
    """Load optimal λ/τ values from Nash parameter selection output."""
    if not os.path.exists(nash_json_path):
        return DEFAULT_LAMBDAS.copy()

    try:
        with open(nash_json_path, 'r') as f:
            nash_data = json.load(f)

        label_to_key = {
            'Binary xQuAD': 'BinaryxQuAD',
            'Smooth xQuAD': 'SmoothxQuAD',
            'FA*IR': 'FAIR',
        }

        fixed_lambdas = {}
        for label, key in label_to_key.items():
            if label in nash_data:
                lambda_vals = [v for v in nash_data[label].values() if v is not None]
                if lambda_vals:
                    from collections import Counter
                    fixed_lambdas[key] = Counter(lambda_vals).most_common(1)[0][0]
                else:
                    fixed_lambdas[key] = DEFAULT_LAMBDAS.get(key, 0.5)
            else:
                fixed_lambdas[key] = DEFAULT_LAMBDAS.get(key, 0.5)

        return fixed_lambdas
    except Exception as e:
        print(f"  [ERROR] Failed to load Nash file: {e}")
        return DEFAULT_LAMBDAS.copy()


class StatisticalTester:
    """Handles all statistical significance testing for recommendation results."""

    ALGOS = ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']

    COLORS = {
        'BinaryxQuAD': '#1f77b4',
        'SmoothxQuAD': '#ff7f0e',
        'FAIR': '#2ca02c',
        'BPR_baseline': '#000000',
    }

    def __init__(self, output_dir: str, nash_json_path: str = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Auto-detect Nash if path not provided
        if nash_json_path is None:
            candidates = [
                os.path.join(output_dir, '..', '..', 'nash_selection', 'nash_optimal_params.json'),
                os.path.join(output_dir, '..', 'nash_selection', 'nash_optimal_params.json'),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    nash_json_path = candidate
                    break

        self.fixed_lambdas = load_nash_lambdas(nash_json_path) if nash_json_path else DEFAULT_LAMBDAS.copy()
        print(f"  StatisticalTester using λ*: {self.fixed_lambdas}")

    def run_all_tests(self, all_per_user: dict, k: int = 10, fixed_lambdas: dict = None,
                      also_run_best_lambda: bool = True):
        lam_map = fixed_lambdas if fixed_lambdas is not None else self.fixed_lambdas

        fixed_rows, legacy_rows = [], []

        for seed, seed_data in all_per_user.items():
            for alpha, alpha_data in seed_data.items():
                baseline_data = alpha_data.get('BPR_baseline', {})
                if not baseline_data:
                    continue
                baseline_users = sorted(baseline_data.keys())
                baseline_vals = np.array([baseline_data[u] for u in baseline_users])

                for algo in self.ALGOS:
                    if algo not in alpha_data:
                        continue

                    target_lam = lam_map.get(algo)
                    if target_lam is not None:
                        fixed_vals = self._get_vals_at_fixed_lambda(alpha_data[algo], baseline_users, target_lam)
                        if fixed_vals is not None:
                            row = self._compute_stats_row(seed, alpha, algo, target_lam, fixed_vals, baseline_vals)
                            row['lambda_mode'] = 'fixed_nash'
                            fixed_rows.append(row)

                    if also_run_best_lambda:
                        best_lam, best_vals = self._find_best_lambda_legacy(alpha_data[algo], baseline_users)
                        if best_vals is not None:
                            row = self._compute_stats_row(seed, alpha, algo, best_lam, best_vals, baseline_vals)
                            row['lambda_mode'] = 'best_per_seed'
                            legacy_rows.append(row)

        fixed_df = pd.DataFrame(fixed_rows)
        summary_df = self._summarise(fixed_df)

        suffix = '_nash_lambda' if lam_map != DEFAULT_LAMBDAS else '_default_lambda'
        fixed_df.to_csv(os.path.join(self.output_dir, f'statistical_tests{suffix}.csv'), index=False)
        summary_df.to_csv(os.path.join(self.output_dir, f'statistical_summary{suffix}.csv'))

        if also_run_best_lambda and legacy_rows:
            legacy_df = pd.DataFrame(legacy_rows)
            legacy_df.to_csv(os.path.join(self.output_dir, 'statistical_tests.csv'), index=False)
            self._summarise(legacy_df).to_csv(os.path.join(self.output_dir, 'statistical_summary.csv'))

        self._plot_significance_heatmap(fixed_df, suffix=suffix)
        self._plot_effect_sizes(fixed_df, suffix=suffix)
        self._plot_win_rates(fixed_df, suffix=suffix)

        if also_run_best_lambda and legacy_rows:
            self._plot_significance_heatmap(legacy_df, suffix='_bestlam')
            self._plot_effect_sizes(legacy_df, suffix='_bestlam')
            self._plot_win_rates(legacy_df, suffix='_bestlam')

        print(f"\n[StatisticalTester] Results saved to {self.output_dir}")
        print(f"  λ* values used: {lam_map}")
        self._print_summary(summary_df)

        return fixed_df, summary_df

    def run_from_per_user_json(self, results_dir: str, seed_prefix: str = 'seed_',
                               json_filename: str = 'per_user_ndcg.json', fixed_lambdas: dict = None, k: int = 10):
        lam_map = fixed_lambdas if fixed_lambdas is not None else self.fixed_lambdas
        seed_dirs = sorted([d for d in os.listdir(results_dir) if
                            d.startswith(seed_prefix) and os.path.isdir(os.path.join(results_dir, d))])

        all_per_user = {}
        for sd in seed_dirs:
            json_path = os.path.join(results_dir, sd, json_filename)
            if not os.path.exists(json_path):
                print(f"  WARNING: {json_path} not found — skipping")
                continue

            with open(json_path) as f:
                raw = json.load(f)

            seed_val = int(sd.replace(seed_prefix, ''))
            seed_data = {}
            for alpha_str, algo_dict in raw.items():
                alpha_f = float(alpha_str)
                seed_data[alpha_f] = {}
                for algo, lam_dict in algo_dict.items():
                    if algo == 'BPR_baseline':
                        seed_data[alpha_f]['BPR_baseline'] = {int(u): v for u, v in lam_dict.items()}
                    else:
                        seed_data[alpha_f][algo] = {float(lam): {int(u): v for u, v in user_dict.items()} for
                                                    lam, user_dict in lam_dict.items()}
            all_per_user[seed_val] = seed_data
            print(f"  Loaded {json_path} ({len(seed_data)} α levels)")

        if not all_per_user:
            raise FileNotFoundError(f"No per-user JSON files found in {results_dir}")

        return self.run_all_tests(all_per_user, k=k, fixed_lambdas=lam_map, also_run_best_lambda=True)

    # Private helper methods (same as before)
    def _get_vals_at_fixed_lambda(self, algo_data, baseline_users, target_lambda):
        if not algo_data:
            return None
        if target_lambda in algo_data:
            user_data = algo_data[target_lambda]
            return np.array([user_data.get(u, 0.0) for u in baseline_users])
        available = list(algo_data.keys())
        closest = min(available, key=lambda l: abs(float(l) - float(target_lambda)))
        user_data = algo_data[closest]
        return np.array([user_data.get(u, 0.0) for u in baseline_users])

    def _find_best_lambda_legacy(self, algo_data, baseline_users):
        best_mean, best_lam, best_vals = -np.inf, None, None
        for lam, user_data in algo_data.items():
            vals = np.array([user_data.get(u, 0.0) for u in baseline_users])
            m = float(np.mean(vals))
            if m > best_mean:
                best_mean, best_lam, best_vals = m, lam, vals
        return best_lam, best_vals

    def _compute_stats_row(self, seed, alpha, algo, lambda_val, intervention_vals, baseline_vals):
        diff = intervention_vals - baseline_vals
        t_stat, p_ttest = ttest_rel(intervention_vals, baseline_vals)
        try:
            _, p_wilcoxon = wilcoxon(intervention_vals, baseline_vals)
        except ValueError:
            p_wilcoxon = 1.0
        std_diff = np.std(diff)
        cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0
        ci_lower, ci_upper = self._bootstrap_ci(diff)
        n = len(baseline_vals)
        wins = int(np.sum(intervention_vals > baseline_vals))
        ties = int(np.sum(np.isclose(intervention_vals, baseline_vals)))
        losses = int(np.sum(intervention_vals < baseline_vals))
        return {
            'seed': seed, 'alpha': float(alpha), 'algorithm': algo, 'best_lambda': lambda_val,
            't_statistic': float(t_stat), 'p_value_ttest': float(p_ttest), 'p_value_wilcoxon': float(p_wilcoxon),
            'cohens_d': cohens_d, 'sig_ttest_0.05': bool(p_ttest < 0.05), 'sig_ttest_0.01': bool(p_ttest < 0.01),
            'sig_wilcoxon_0.05': bool(p_wilcoxon < 0.05), 'mean_improvement': float(np.mean(diff)),
            'std_improvement': float(np.std(diff)), 'ci_lower_95': float(ci_lower), 'ci_upper_95': float(ci_upper),
            'wins': wins, 'ties': ties, 'losses': losses, 'total_users': n, 'win_rate': wins / n,
        }

    def _summarise(self, stats_df):
        return stats_df.groupby(['alpha', 'algorithm']).agg({
            'p_value_ttest': ['mean', 'std', 'min', 'max'], 'cohens_d': ['mean', 'std'],
            'sig_ttest_0.05': 'mean', 'mean_improvement': ['mean', 'std'], 'win_rate': ['mean', 'std'],
        }).round(4)

    def _bootstrap_ci(self, diff, n_bootstrap=1000, ci=0.95):
        means = [np.mean(np.random.choice(diff, size=len(diff), replace=True)) for _ in range(n_bootstrap)]
        return (float(np.percentile(means, (1 - ci) / 2 * 100)), float(np.percentile(means, (1 + ci) / 2 * 100)))

    def _print_summary(self, summary_df):
        print("\n  ── Fixed-λ Significance Summary ──")
        try:
            sig_col = ('sig_ttest_0.05', 'mean')
            cd_col = ('cohens_d', 'mean')
            for (alpha, algo), row in summary_df.iterrows():
                print(f"  α={alpha}  {algo:<14}  sig_rate={row[sig_col]:.1%}  d={row[cd_col]:.4f}")
        except Exception:
            print(summary_df.to_string())

    def _plot_significance_heatmap(self, stats_df, suffix=''):
        alphas = sorted(stats_df['alpha'].unique())
        sig_matrix = np.zeros((len(self.ALGOS), len(alphas)))
        for i, algo in enumerate(self.ALGOS):
            for j, alpha in enumerate(alphas):
                sub = stats_df[(stats_df['algorithm'] == algo) & (stats_df['alpha'] == alpha)]
                if not sub.empty:
                    sig_matrix[i, j] = sub['sig_ttest_0.05'].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(sig_matrix, annot=True, fmt='.0%', xticklabels=[f'α={a}' for a in alphas],
                    yticklabels=self.ALGOS, cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                    cbar_kws={'label': 'Proportion of seeds p<0.05 (paired t-test)'})
        mode_label = 'Nash λ*' if 'nash' in suffix else ('Fixed λ*' if 'fixed' in suffix else 'Best λ per seed')
        ax.set_title(f'Statistical Significance Heatmap ({mode_label})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'significance_heatmap{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_effect_sizes(self, stats_df, suffix=''):
        alphas = sorted(stats_df['alpha'].unique())
        x, w = np.arange(len(alphas)), 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, algo in enumerate(self.ALGOS):
            means, errs = [], []
            for alpha in alphas:
                s = stats_df[(stats_df['algorithm'] == algo) & (stats_df['alpha'] == alpha)]
                means.append(s['cohens_d'].mean() if not s.empty else 0)
                errs.append(s['cohens_d'].std() if not s.empty else 0)
            ax.bar(x + (i - 1) * w, means, w, label=algo, color=self.COLORS[algo], alpha=0.8, yerr=errs, capsize=5)
        for y, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
            ax.axhline(y=y, color='gray', linestyle='--', alpha=0.4, label=f'{label} effect')
        mode_label = 'Nash λ*' if 'nash' in suffix else ('Fixed λ*' if 'fixed' in suffix else 'Best λ per seed')
        ax.set_xlabel('Bias Level (α)'), ax.set_ylabel("Cohen's d")
        ax.set_title(f"Effect Size of Fairness Interventions vs Baseline ({mode_label})", fontsize=14,
                     fontweight='bold')
        ax.set_xticks(x), ax.set_xticklabels([f'α={a}' for a in alphas])
        ax.legend(loc='best'), ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'effect_sizes{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_win_rates(self, stats_df, suffix=''):
        alphas = sorted(stats_df['alpha'].unique())
        x, w = np.arange(len(alphas)), 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, algo in enumerate(self.ALGOS):
            wrs = []
            for alpha in alphas:
                s = stats_df[(stats_df['algorithm'] == algo) & (stats_df['alpha'] == alpha)]
                wrs.append(s['win_rate'].mean() if not s.empty else 0)
            ax.bar(x + (i - 1) * w, wrs, w, label=algo, color=self.COLORS[algo], alpha=0.8)
        ax.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)', alpha=0.7)
        mode_label = 'Nash λ*' if 'nash' in suffix else ('Fixed λ*' if 'fixed' in suffix else 'Best λ per seed')
        ax.set_xlabel('Bias Level (α)'), ax.set_ylabel('Win Rate vs Baseline')
        ax.set_title(f'Proportion of Users Benefiting from Fairness Intervention ({mode_label})', fontsize=14,
                     fontweight='bold')
        ax.set_xticks(x), ax.set_xticklabels([f'α={a}' for a in alphas])
        ax.legend(loc='best'), ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'win_rates{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run statistical significance tests')
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--nash-json', type=str, default=None)
    parser.add_argument('--mode', type=str, default='json', choices=['csv', 'json'])
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, 'enhanced_analysis', 'statistics')
    tester = StatisticalTester(out_dir, nash_json_path=args.nash_json)

    if args.mode == 'csv':
        tester.run_from_csvs(args.results_dir, k=args.k)
    else:
        tester.run_from_per_user_json(args.results_dir, k=args.k)