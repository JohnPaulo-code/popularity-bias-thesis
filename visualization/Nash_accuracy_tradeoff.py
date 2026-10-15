"""
Table — Nash / Balance score per method per alpha (to accompany the
Pareto frontier, Figure fig:pareto_frontier).
=====================================================================
One value per (method, alpha) summarising the APLT–NDCG balance, so the
Pareto frontier gets a quantitative companion table.

Two score variants are emitted because they tell different stories:

  RAW NASH (Eq. 4.1, same as tab_nash_selection):
      nash = (NDCG/NDCG_base) * (APLT/APLT_base)
    - Baseline == 1.000 by construction.
    - RISES with alpha because APLT_base -> 0 inflates the ratio
      (the "relative headroom" effect). Safe to read DOWN a column
      (method comparison at fixed alpha); misleading ACROSS a row.

  ABSOLUTE BALANCE (geometric mean of the two raw metrics):
      gbal = sqrt(NDCG * APLT)
    - Falls with alpha, matching the structural ceiling.
    - Safe to read in BOTH directions. Baseline is NOT 1.0; it is just
      another operating point, which is arguably more honest for a
      Pareto companion.

Pick one for the main text; the other can go in an appendix or be
dropped. The recommendation is to lead with absolute balance in Section
4.3 (it reinforces "ceiling falls") and, if desired, keep raw Nash in
the appendix next to tab_nash_selection.

Mirrors the data-loading of the Figure-4 / Pareto script so the numbers
match the plotted points exactly.
"""

import os
import json
import numpy as np
import pandas as pd

BASE = r"D:\Final_thesis\thesis_project"

BPR_JSONS = {
    'MovieLens':   os.path.join(BASE, r"sensitivity_analysis\sa_results_ml_v1\enhanced_analysis\aggregated_results.json"),
    'AmazonBooks': os.path.join(BASE, r"sensitivity_analysis\sa_results_ab_v1\enhanced_analysis\aggregated_results.json"),
}
OUTPUT_DIR   = os.path.join(BASE, r"figures")
ALPHA_LEVELS = ['1.0', '1.2', '1.5', '1.75', '2.0']
FIXED = {'BinaryxQuAD': {'MovieLens': 0.4, 'AmazonBooks': 0.3},
         'SmoothxQuAD': {'MovieLens': 0.9, 'AmazonBooks': 0.9},
         'FAIR':        {'MovieLens': 0.9, 'AmazonBooks': 0.9}}

METHOD_ORDER  = ['FAIR', 'SmoothxQuAD', 'BinaryxQuAD']   # FA*IR first (best)
METHOD_LABELS = {'FAIR': 'FA*IR', 'SmoothxQuAD': 'Smooth xQuAD',
                 'BinaryxQuAD': 'Binary xQuAD', 'Baseline': 'Baseline'}
DS_LABELS = {'MovieLens': 'MovieLens-1M', 'AmazonBooks': 'Amazon Books'}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_idx(grid, target):
    rounded = [round(g, 4) for g in grid]
    try:    return rounded.index(round(target, 4))
    except ValueError:
        return int(np.argmin(np.abs(np.array(grid) - target)))


def load_bpr(json_path, dataset):
    """Return data[alpha][method] = {'ndcg','aplt'} at Nash-fixed params."""
    with open(json_path) as f:
        raw = json.load(f)
    lambda_grid = raw['lambda_grid']
    tau_grid    = raw.get('tau_grid', [round(x * 0.1, 1) for x in range(11)])
    data = {}
    for a in raw['alphas']:
        if a not in raw['BPR_baseline']:
            continue
        row = {'Baseline': {
            'ndcg': float(raw['BPR_baseline'][a]['NDCG@10']['mean']),
            'aplt': float(raw['BPR_baseline'][a]['APLT@10']['mean'])}}
        for method in ['BinaryxQuAD', 'SmoothxQuAD', 'FAIR']:
            if method not in raw:
                continue
            grid = tau_grid if method == 'FAIR' else lambda_grid
            idx  = get_idx(grid, FIXED[method][dataset])
            nd   = raw[method][a]['NDCG@10']['mean']
            ap   = raw[method][a]['APLT@10']['mean']
            row[method] = {
                'ndcg': float(nd[idx] if isinstance(nd, list) else nd),
                'aplt': float(ap[idx] if isinstance(ap, list) else ap)}
        data[a] = row
    return data


def compute_scores(data):
    """For each alpha, method: raw Nash and absolute geometric balance."""
    out = {}
    for a in ALPHA_LEVELS:
        if a not in data:
            continue
        base = data[a]['Baseline']
        out[a] = {}
        for method, vals in data[a].items():
            nash = ((vals['ndcg'] / base['ndcg']) *
                    (vals['aplt'] / base['aplt'])) if base['ndcg'] > 0 and base['aplt'] > 0 else float('nan')
            gbal = float(np.sqrt(max(vals['ndcg'], 0) * max(vals['aplt'], 0)))
            out[a][method] = {'nash': nash, 'gbal': gbal}
    return out


def emit_latex(scores_ml, scores_ab, score_key, caption, label, fmt, outpath,
               include_baseline=False):
    methods = METHOD_ORDER + (['Baseline'] if include_baseline else [])
    L = []
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(r'\caption{' + caption + r'}')
    L.append(r'\label{' + label + r'}')
    L.append(r'\begin{tabular}{ll' + 'c' * len(ALPHA_LEVELS) + r'}')
    L.append(r'\toprule')
    L.append(r'Dataset & Method & ' +
             ' & '.join([r'$\alpha=' + a + r'$' for a in ALPHA_LEVELS]) + r' \\')
    L.append(r'\midrule')
    for ds_i, (ds, scores) in enumerate([('MovieLens', scores_ml),
                                         ('AmazonBooks', scores_ab)]):
        for m_i, method in enumerate(methods):
            shade = r'\rowcolor{gray!6} ' if (m_i % 2 == 1) else ''
            ds_cell = r'\multirow{' + str(len(methods)) + r'}{*}{' + \
                      DS_LABELS[ds] + r'}' if m_i == 0 else ''
            # collect values, bold the best (max) per column among methods
            row_vals = []
            for a in ALPHA_LEVELS:
                v = scores.get(a, {}).get(method, {}).get(score_key, float('nan'))
                row_vals.append(v)
            cells = []
            for j, a in enumerate(ALPHA_LEVELS):
                col_vals = [scores.get(a, {}).get(mm, {}).get(score_key, float('nan'))
                            for mm in METHOD_ORDER]  # best among real methods only
                v = row_vals[j]
                txt = fmt(v)
                if method in METHOD_ORDER and not np.isnan(v) and \
                   v == np.nanmax(col_vals):
                    txt = r'\textbf{' + txt + r'}'
                cells.append(txt)
            L.append(f"{shade}{ds_cell} & {METHOD_LABELS[method]} & " +
                     ' & '.join(cells) + r' \\')
        if ds_i == 0:
            L.append(r'\midrule')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"  LaTeX -> {outpath}")


def print_console(scores_ml, scores_ab, key, title):
    for ds, scores in [('MovieLens-1M', scores_ml), ('Amazon Books', scores_ab)]:
        print(f"\n  {title} — {ds}")
        print(f"    {'Method':>14}  " +
              '  '.join([f'a={a:>4}' for a in ALPHA_LEVELS]))
        for method in METHOD_ORDER + ['Baseline']:
            row = f"    {METHOD_LABELS[method]:>14}  "
            for a in ALPHA_LEVELS:
                v = scores.get(a, {}).get(method, {}).get(key, float('nan'))
                row += f"  {v:>5.3f}" if not np.isnan(v) else f"  {'n/a':>5}"
            print(row)


def main():
    print("=" * 60)
    print("  Nash / Balance companion table for the Pareto frontier")
    print("=" * 60)
    data_ml = load_bpr(BPR_JSONS['MovieLens'], 'MovieLens')
    data_ab = load_bpr(BPR_JSONS['AmazonBooks'], 'AmazonBooks')
    s_ml = compute_scores(data_ml)
    s_ab = compute_scores(data_ab)

    print_console(s_ml, s_ab, 'nash', 'RAW NASH (rises with alpha)')
    print_console(s_ml, s_ab, 'gbal', 'ABS GEOMETRIC BALANCE (falls with alpha)')

    # Variant A — raw Nash (matches Eq. 4.1)
    emit_latex(
        s_ml, s_ab, 'nash',
        caption=(r'Nash score (Eq.~\ref{eq:nash_score}), '
                 r'$(\mathrm{NDCG}/\mathrm{NDCG}_{\mathrm{base}})\times'
                 r'(\mathrm{APLT}/\mathrm{APLT}_{\mathrm{base}})$, at '
                 r'Nash-selected parameters for BPR. The best method per '
                 r'column is bolded. Values rise with $\alpha$ because '
                 r'they are measured relative to a collapsing baseline '
                 r'(see Section~\ref{sec:nash_selection}); compare methods '
                 r'down each column, not across rows.'),
        label='tab:pareto_nash',
        fmt=lambda v: f"{v:.2f}" if not np.isnan(v) else 'n/a',
        outpath=os.path.join(OUTPUT_DIR, 'tab_pareto_nash.tex'))

    # Variant B — absolute geometric balance (recommended for main text)
    emit_latex(
        s_ml, s_ab, 'gbal',
        caption=(r'Absolute balance score $\sqrt{\mathrm{NDCG@10}\times'
                 r'\mathrm{APLT@10}}$ at Nash-selected parameters for BPR, '
                 r'a single summary of each method''s position on the '
                 r'accuracy--fairness frontier (Figure~\ref{fig:pareto_frontier}). '
                 r'Unlike the relative Nash score, this absolute measure '
                 r'declines with $\alpha$, tracking the structural ceiling. '
                 r'The best method per column is bolded.'),
        label='tab:pareto_balance',
        fmt=lambda v: f"{v:.3f}" if not np.isnan(v) else 'n/a',
        outpath=os.path.join(OUTPUT_DIR, 'tab_pareto_balance.tex'),
        include_baseline=True)

    print("\nDone.")


if __name__ == "__main__":
    main()