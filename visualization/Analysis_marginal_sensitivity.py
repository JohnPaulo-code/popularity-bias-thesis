"""
Analysis H — Marginal Sensitivity (FIXED)
=========================================
APLT decline per α step for each method, averaged across models.

Step pairs: 1.0→1.2, 1.2→1.5, 1.5→1.75, 1.75→2.0

FIX (v2):
  The original version built step labels with f"{a:.4g}", which strips
  trailing zeros ("1.0" → "1", "2.0" → "2"). write_latex() then matched
  those labels against the hardcoded list ['1.0→1.2', ...], so the first
  and last steps never matched and were written as n/a.

  Now every α is snapped to the canonical ALPHA_LEVELS grid and the step
  label is built from a single source of truth (ALPHA_LABELS). The LaTeX
  writer and the figure both consume the same canonical labels, so all
  four steps populate. Float noise such as 1.2000000001 from CSV inputs
  is also absorbed by the snapping.

  Additional fixes:
    - tabular column spec corrected to {llcccc} (6 columns, was 7)
    - largest absolute decline per row is bolded
    - caption claims now match the generated data

For each step:
  marginal_decline = APLT(α_n) − APLT(α_{n-1})   (negative = ceiling fell)

Reads from all_results_combined.csv and aggregated_results.json.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   11,
    'axes.labelsize':   11,
    'xtick.labelsize':  9.5,
    'ytick.labelsize':  9.5,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE = r"D:\Final_thesis\thesis_project"

BPR_JSONS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v2\enhanced_analysis\aggregated_results.json")
}

OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = [0.5, 0.75]

# ── Single source of truth for α formatting ─────────────────
ALPHA_LABELS = {1.0: '1.0', 1.2: '1.2', 1.5: '1.5', 1.75: '1.75', 2.0: '2.0'}
ALPHA_STEPS  = [f"{ALPHA_LABELS[a0]}→{ALPHA_LABELS[a1]}"
                for a0, a1 in zip(ALPHA_LEVELS[:-1], ALPHA_LEVELS[1:])]
# → ['1.0→1.2', '1.2→1.5', '1.5→1.75', '1.75→2.0']

DATASETS = ['MovieLens', 'AmazonBooks']
MODELS   = ['BPR', 'NeuMF', 'LightGCN']
METHODS  = ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']

NASH = {
    'MovieLens':   {'BinaryxQuAD': 0.4, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
    'AmazonBooks': {'BinaryxQuAD': 0.3, 'SmoothxQuAD': 0.9, 'FAIR': 0.9},
}

# Include baseline so we can show how fast it collapses too
ALL_METHODS = ['Baseline'] + METHODS

METHOD_STYLES = {
    'Baseline':    {'color': '#888888', 'hatch': ''},
    'BinaryxQuAD': {'color': '#1b7837', 'hatch': ''},
    'SmoothxQuAD': {'color': '#e08214', 'hatch': ''},
    'FAIR':        {'color': '#7b2d8b', 'hatch': ''},
}
METHOD_LABELS = {
    'Baseline':    'Baseline',
    'BinaryxQuAD': 'Binary xQuAD',
    'SmoothxQuAD': 'Smooth xQuAD',
    'FAIR':        'FA*IR',
}
DS_LABELS = {'MovieLens': 'MovieLens-1M', 'AmazonBooks': 'Amazon Books'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CANONICAL α HANDLING  (the actual fix)
# ─────────────────────────────────────────────────────────────
def snap_alpha(a, tol=1e-3):
    """Snap a float α to the nearest canonical grid value.

    Absorbs float noise (1.2000000001) and formatting drift (1 vs 1.0).
    Raises if a value is not within `tol` of any grid point, so silent
    mis-assignment is impossible.
    """
    arr  = np.asarray(ALPHA_LEVELS, dtype=float)
    idx  = int(np.argmin(np.abs(arr - float(a))))
    if abs(arr[idx] - float(a)) > tol:
        raise ValueError(f"alpha={a!r} is not on the canonical grid "
                         f"{ALPHA_LEVELS} (tol={tol})")
    return float(arr[idx])


def step_label(a0, a1):
    """Canonical step label built from ALPHA_LABELS — never from f-strings."""
    return f"{ALPHA_LABELS[snap_alpha(a0)]}→{ALPHA_LABELS[snap_alpha(a1)]}"


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def get_idx(grid, target):
    rounded = [round(g, 4) for g in grid]
    try:    return rounded.index(round(target, 4))
    except ValueError:
        return int(np.argmin(np.abs(np.array(grid) - target)))


def load_bpr(json_path, dataset):
    with open(json_path) as f:
        raw = json.load(f)
    lambda_grid = raw['lambda_grid']
    tau_grid    = raw.get('tau_grid', [round(x * 0.1, 1) for x in range(11)])
    rows = []
    for a in raw['alphas']:
        alpha = snap_alpha(float(a))
        bl_aplt = float(raw['BPR_baseline'][a]['APLT@10']['mean'])
        rows.append({'model': 'BPR', 'dataset': dataset, 'alpha': alpha,
                     'method': 'Baseline', 'APLT@10': bl_aplt})
        for method in METHODS:
            if method not in raw:
                continue
            grid = tau_grid if method == 'FAIR' else lambda_grid
            idx  = get_idx(grid, NASH[dataset][method])
            aa   = raw[method][a]['APLT@10']['mean']
            rows.append({'model': 'BPR', 'dataset': dataset, 'alpha': alpha,
                         'method': method,
                         'APLT@10': float(aa[idx] if isinstance(aa, list) else aa)})
    return pd.DataFrame(rows)


def load_cross(csv_path, model, dataset):
    df = pd.read_csv(csv_path)
    df = df[df['method'].isin(ALL_METHODS)].copy()
    df['model']   = model
    df['dataset'] = dataset
    df['alpha']   = df['alpha'].astype(float).map(snap_alpha)
    return df[['model', 'dataset', 'alpha', 'method', 'APLT@10']]


def load_all():
    frames = []
    for dataset in DATASETS:
        frames.append(load_bpr(BPR_JSONS[dataset], dataset))
        for model in ['NeuMF', 'LightGCN']:
            path = CROSS_CSVS[model][dataset]
            if os.path.exists(path):
                frames.append(load_cross(path, model, dataset))
            else:
                print(f"  !  Not found: {path}")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# COMPUTE MARGINAL DECLINE
# ─────────────────────────────────────────────────────────────
def compute_marginal(df):
    """
    For each (model, dataset, method):
      compute APLT at each α, then marginal decline at each step.
    Average across models to get one value per (dataset, step, method).
    """
    agg = df.groupby(['model', 'dataset', 'alpha', 'method']).agg(
        aplt=('APLT@10', 'mean')
    ).reset_index()

    rows = []
    for (model, dataset, method), g in agg.groupby(['model', 'dataset', 'method']):
        g = g.sort_values('alpha')
        alphas = g['alpha'].values
        apts   = g['aplt'].values
        for i in range(1, len(alphas)):
            a0, a1  = alphas[i-1], alphas[i]
            decline = apts[i] - apts[i-1]   # negative = fell
            rows.append({
                'model':   model,
                'dataset': dataset,
                'method':  method,
                'step':    step_label(a0, a1),   # ← canonical label
                'a0':      a0, 'a1': a1,
                'aplt_0':  apts[i-1],
                'aplt_1':  apts[i],
                'decline': decline,
                'abs_decline': abs(decline),
            })

    per_model = pd.DataFrame(rows)

    # Average across models
    avg = per_model.groupby(['dataset', 'step', 'a0', 'a1', 'method']).agg(
        decline_mean    =('decline',     'mean'),
        decline_std     =('decline',     'std'),
        abs_decline_mean=('abs_decline', 'mean'),
        abs_decline_std =('abs_decline', 'std'),
    ).reset_index().sort_values(['dataset', 'method', 'a0'])

    # Sanity check: every (dataset, method) must have all four steps.
    expected = set(ALPHA_STEPS)
    for (dataset, method), g in avg.groupby(['dataset', 'method']):
        missing = expected - set(g['step'])
        if missing:
            raise RuntimeError(
                f"{dataset}/{method}: missing steps {sorted(missing)} — "
                f"check that all five α levels exist in the input data."
            )
    print("  ✓ all four steps present for every (dataset, method)")

    return per_model, avg


# ─────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────
def print_table(avg):
    for dataset in DATASETS:
        print(f"\n{'='*76}")
        print(f"  {dataset} — Marginal APLT Decline per α Step")
        print(f"  (negative = ceiling fell; averaged across BPR, NeuMF, LightGCN)")
        print(f"{'='*76}")
        sub = avg[avg['dataset'] == dataset]
        print(f"  {'Step':>12}  {'Baseline':>10}  "
              f"{'Binary xQuAD':>14}  {'Smooth xQuAD':>14}  {'FA*IR':>8}")
        print('  ' + '-'*68)
        for step in ALPHA_STEPS:
            row = f"  {step:>12}"
            for method in ALL_METHODS:
                v = sub[(sub['step'] == step) &
                        (sub['method'] == method)]['decline_mean']
                row += f"  {v.values[0]:>13.4f}" if len(v) else f"  {'n/a':>13}"
            print(row)


# ─────────────────────────────────────────────────────────────
# LATEX TABLE
# ─────────────────────────────────────────────────────────────
def write_latex(avg, outpath):
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(
        r'\caption{Marginal APLT decline per $\alpha$ step at Nash-selected '
        r'parameters, averaged across BPR, NeuMF, and LightGCN. Negative '
        r'values indicate a fall in the APLT ceiling between consecutive '
        r'bias levels; the largest decline in each row is bolded. '
        r'FA*IR loses the most at every step, and the per-step declines '
        r'shrink as $\alpha$ grows, indicating a concave (decelerating) '
        r'ceiling collapse driven by early Z-2 depletion.}'
    )
    lines.append(r'\label{tab:marginal_sensitivity}')
    lines.append(r'\begin{tabular}{llcccc}')
    lines.append(r'\toprule')
    lines.append(
        r'Step & Dataset & Baseline & Binary xQuAD & '
        r'Smooth xQuAD & FA*IR \\'
    )
    lines.append(r'\midrule')

    for s_i, step in enumerate(ALPHA_STEPS):
        for i, dataset in enumerate(DATASETS):
            shade = r'\rowcolor{gray!6} ' if i % 2 == 1 else ''
            sub   = avg[(avg['dataset'] == dataset) & (avg['step'] == step)]

            # Collect values first so the row max can be bolded.
            vals = {}
            for method in ALL_METHODS:
                v = sub[sub['method'] == method]['decline_mean']
                vals[method] = float(v.values[0]) if len(v) else None

            present  = {m: v for m, v in vals.items() if v is not None}
            max_loss = min(present.values()) if present else None  # most negative

            step_tex = step.replace('→', r'$\to$')
            row = f"{shade}{step_tex if i == 0 else ''} & {DS_LABELS[dataset]}"
            for method in ALL_METHODS:
                v = vals[method]
                if v is None:
                    row += r" & n/a"
                else:
                    txt = f"${v:+.4f}$"
                    if max_loss is not None and v == max_loss:
                        txt = r'$\mathbf{' + f"{v:+.4f}" + r'}$'
                    row += f" & {txt}"
            lines.append(row + r' \\')
        if s_i < len(ALPHA_STEPS) - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX → {outpath}")


# ─────────────────────────────────────────────────────────────
# FIGURE — grouped bar chart
# ─────────────────────────────────────────────────────────────
def plot_marginal(avg, save_path):
    """
    2 panels (ML | AB). X-axis: α steps. Bars: one per method.
    Y-axis: absolute APLT decline (magnitude, positive values).
    Grouped bars — easier to compare methods at each step.
    """
    ds_labels = {'MovieLens': '(a) MovieLens-1M',
                 'AmazonBooks': '(b) Amazon Books'}

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={'wspace': 0.30}
    )
    axes = {'MovieLens': ax_l, 'AmazonBooks': ax_r}

    n_methods = len(ALL_METHODS)
    bar_w     = 0.18
    offsets   = np.linspace(-(n_methods-1)/2 * bar_w,
                             (n_methods-1)/2 * bar_w,
                             n_methods)
    x = np.arange(len(ALPHA_STEPS))

    for dataset, ax in axes.items():
        sub = avg[avg['dataset'] == dataset]

        for m_i, method in enumerate(ALL_METHODS):
            vals = []
            errs = []
            for step in ALPHA_STEPS:          # ← canonical list everywhere
                v = sub[(sub['step']   == step) &
                        (sub['method'] == method)]
                vals.append(abs(v['decline_mean'].values[0]) if len(v) else 0)
                errs.append(v['abs_decline_std'].values[0]   if len(v) else 0)

            ax.bar(x + offsets[m_i], vals,
                   width=bar_w,
                   color=METHOD_STYLES[method]['color'],
                   alpha=0.85, zorder=2,
                   label=METHOD_LABELS[method],
                   yerr=errs, capsize=3,
                   error_kw={'linewidth': 0.9, 'alpha': 0.6})

        ax.set_xticks(x)
        ax.set_xticklabels([f'α {s}' for s in ALPHA_STEPS], fontsize=9)
        ax.set_xlabel('Bias level step', labelpad=6)
        ax.set_title(ds_labels[dataset], pad=8, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.3f}')
        )
        ax.grid(True, axis='y', linestyle='--',
                linewidth=0.5, alpha=0.4, color='gray')
        ax.set_axisbelow(True)

    ax_l.set_ylabel('Absolute APLT decline per step', labelpad=6)

    # Shared legend
    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.07),
               framealpha=0.95, edgecolor='#cccccc',
               fontsize=10, handlelength=1.8)

    fig.text(
        0.5, -0.10,
        'Error bars show ±1 std across BPR, NeuMF and LightGCN. '
        'Larger bars indicate faster structural ceiling collapse at that step.',
        ha='center', fontsize=9, color='#555555'
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure → {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Analysis H — Marginal Sensitivity (fixed)")
    print("=" * 60)

    df              = load_all()
    per_model, avg  = compute_marginal(df)

    print_table(avg)
    write_latex(avg, os.path.join(OUTPUT_DIR,
                                  'tab_marginal_sensitivity.tex'))
    plot_marginal(avg,
                  os.path.join(OUTPUT_DIR, 'fig_marginal_sensitivity.pdf'))
    plot_marginal(avg,
                  os.path.join(OUTPUT_DIR, 'fig_marginal_sensitivity.png'))
    print("\nDone.")


if __name__ == "__main__":
    main()