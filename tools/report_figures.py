"""Rebuild report artifacts from immutable saved results, without optimization or retrieval.

Run from the project root with a Python containing numpy and matplotlib:
  python tools/report_figures.py
Outputs go to artifacts/report/ unless --output names another directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/comp6704-report-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = [0.01, 0.001, 0.0001, 0.00001]
METHODS = ['gd', 'newton', 'adam']
LABELS = {'gd': 'GD', 'newton': 'Newton', 'adam': 'Adam', 'unit': 'Unit-trial GD'}
# Okabe-Ito colors, with line styles retained for monochrome reproduction.
PALETTE = {'blue': '#0072B2', 'vermillion': '#D55E00', 'green': '#009E73',
           'purple': '#CC79A7', 'sky': '#56B4E9', 'orange': '#E69F00'}
STYLES = {'gd': (PALETTE['blue'], '-'), 'newton': (PALETTE['vermillion'], '--'),
          'adam': (PALETTE['green'], '-.'), 'unit': (PALETTE['purple'], ':')}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def target(run, seed, lam, n=5936):
    return run / 'targets' / f'seed{seed}-lambda{lam:g}-n{n}'


def fmt(x, digits=4):
    return f'{x:.{digits}f}'


def ltex(lam):
    return r'$10^{%d}$' % int(round(np.log10(lam)))


def configure():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': 'white', 'lines.linewidth': 1.2})


def finish(fig, output, stem):
    fig.savefig(output / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(output / f'{stem}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def fig1(run, calibration, out):
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 5.1), layout='constrained')
    for row, lam in enumerate(LAMBDAS):
        for method in METHODS + ['unit']:
            path = calibration / f'lambda-{lam:g}' / 'gd-unit.json' if method == 'unit' else target(run, 42, lam) / f'{method}-trace.json'
            h = read(path)['history']
            y = np.maximum([x['certificate_B'] for x in h], np.finfo(float).tiny)
            color, style = STYLES[method]
            axes[row, 0].semilogy([x['iteration'] for x in h], y, color=color, ls=style, label=LABELS[method])
            axes[row, 1].semilogy([x['seconds'] * 1000 for x in h], y, color=color, ls=style, label=LABELS[method])
        for col in range(2):
            ax = axes[row, col]
            ax.axhline(min(1e-9, .5e-6*lam), color='0.6', lw=.8, ls=(0, (1, 3)))
            ax.set_title(r'$\lambda=10^{%d}$' % int(round(np.log10(lam))))
            if col == 0:
                ax.set_ylabel(r'$B(a)$')
            # Label every fifth decade so the start level and threshold can be read.
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10.0**k for k in range(-30, 6, 5)]))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.grid(axis='y', alpha=.18)
        axes[row, 0].set_xscale('symlog', linthresh=1)
        axes[row, 0].set_xlim(left=0)
        if row == 3:
            axes[row, 0].set_xlabel('Updates (symmetric log)')
        axes[row, 1].set_xscale('symlog', linthresh=1)
        axes[row, 1].set_xlim(left=0)
        if row == 3:
            axes[row, 1].set_xlabel('Trace time (ms, symmetric log)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False)
    finish(fig, out, 'fig1-convergence')


def time_groups(e1):
    groups = []
    for lam in LAMBDAS:
        for method in METHODS:
            targets = [x for x in e1['targets'] if x['regularization'] == lam]
            medians = [x['timing'][method]['median_seconds'] for x in targets]
            iqrs = [x['timing'][method]['iqr_seconds'] for x in targets]
            iterations = [next(c['iterations'] for c in x['checks'] if c['method'] == method) for x in targets]
            groups.append({'regularization': lam, 'method': method,
                'seed_medians_seconds': medians, 'seed_iqrs_seconds': iqrs,
                'median_of_seed_medians_seconds': float(np.median(medians)),
                'seed_median_range_seconds': [min(medians), max(medians)],
                'within_seed_iqr_range_seconds': [min(iqrs), max(iqrs)],
                'iterations_range': [min(iterations), max(iterations)],
                'successes': sum(all(c['status'] == 'converged' for c in x['checks'] if c['method'] == method) for x in targets),
                'timing_complete': sum(x['timing'][method]['complete'] for x in targets)})
    return groups


def fig2(run, groups, out, data):
    fig = plt.figure(figsize=(5.5, 3.6), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    ax = axes[0]
    for method in METHODS:
        g = [x for x in groups if x['method'] == method]
        y = np.array([x['median_of_seed_medians_seconds'] for x in g]) * 1000
        low = np.array([x['seed_median_range_seconds'][0] for x in g]) * 1000
        high = np.array([x['seed_median_range_seconds'][1] for x in g]) * 1000
        color, style = STYLES[method]
        ax.errorbar(LAMBDAS, y, yerr=[y-low, high-y], label=LABELS[method], color=color, ls=style, marker='o', ms=4, capsize=3)
    ax.set(xscale='log', yscale='log', xlabel=r'$\lambda$', ylabel='Solve time (ms)', title='(a) Independent solve timing')
    spectra = []
    for lam in LAMBDAS:
        d = read(target(run, 42, lam) / 'diagnostics.json')
        spectra.append({'regularization': lam, 'initial_condition': d['initial_spectrum']['full_condition'],
            'reference_condition': d['reference_spectrum']['full_condition'],
            'global_condition_bound': (lam+d['initial_spectrum']['global_curvature_bound'])/lam})
    for key, label, ls, color in [('initial_condition', 'Initial', '-', '0.6'), ('reference_condition', 'Refined Newton', '--', '0.15'), ('global_condition_bound', 'Global bound', ':', '0.35')]:
        axes[1].plot(LAMBDAS, [x[key] for x in spectra], label=label, ls=ls, color=color, marker='o', ms=4)
    axes[1].set(xscale='log', yscale='log', xlabel=r'$\lambda$', ylabel='Full-space condition number', title='(b) Curvature, seed 42')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.14), ncol=3, columnspacing=.7, handlelength=1.5, frameon=False)
    axes[1].legend(loc='lower left', bbox_to_anchor=(0, 1.14), ncol=2, columnspacing=.6, handlelength=1.3, frameon=False)
    mb = read(target(run, 42, .001) / 'microbench.json')
    proxy = []
    for method in METHODS:
        counts = read(target(run, 42, .001) / f'{method}-trace.json')['costs']
        proxy.append({'method': method, 'Xv': counts['Xv'], 'XTv': counts['XTv'],
            'hessian_builds': counts['hessian_builds'], 'linear_solves': counts['linear_solves'],
            'matvec_proxy_seconds': (counts['Xv']+counts['XTv'])/2*mb['gradient']['median_seconds'],
            'hessian_proxy_seconds': counts['hessian_builds']*mb['hessian']['median_seconds'],
            'linear_proxy_seconds': counts['linear_solves']*mb['linear_solve']['median_seconds']})
    bottom = np.zeros(3)
    for key, label, hatch, shade in [('matvec_proxy_seconds', 'Matvec proxy', '', '0.75'), ('hessian_proxy_seconds', 'Hessian proxy', '///', '0.75'), ('linear_proxy_seconds', 'Linear solve proxy', '...', '0.75')]:
        heights = np.array([x[key] for x in proxy])*1000
        axes[2].bar([LABELS[x] for x in METHODS], heights, bottom=bottom, label=label, color=shade, edgecolor='black', hatch=hatch, linewidth=.6)
        bottom += heights
    axes[2].set(ylabel='Cost proxy (ms)', title=r'(c) Cost proxy, seed 42, $\lambda=10^{-3}$')
    axes[2].legend(loc='upper left')
    data['spectra'] = spectra
    data['cost_proxy'] = {'records': proxy, 'microbenchmark_medians_seconds': {k:v['median_seconds'] for k,v in mb.items()},
        'formula': '(Xv+XTv)/2 * gradient_median + hessian_builds*hessian_median + linear_solves*linear_solve_median',
        'units': 'seconds in data; milliseconds in figure',
        'limitations': 'Equal matvec cost assumed; gradient includes non-matrix work, Hessian includes margin computation. Not an identifiable exact decomposition and not measured wall time.'}
    finish(fig, out, 'fig2-regularization-cost')


def fig3(run, e1, e3, out, data):
    sizes = [512, 2048, 5936]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), layout='constrained')
    micro = []
    for n in sizes:
        mb = read(target(run, 42, .001, n) / 'microbench.json')
        micro.append({'n': n, **{k:v['median_seconds'] for k,v in mb.items()}})
    for op, label, ls, marker in [('gradient', 'Value + gradient', '-', 'o'), ('hessian', 'Hessian', '--', 's'), ('linear_solve', 'Linear solve', ':', '^')]:
        axes[0].plot(sizes, [x[op]*1000 for x in micro], marker=marker, ls=ls, color='0.35', label=label)
    axes[0].set(xscale='log', yscale='log', xlabel='Training triples N', ylabel='Operation time (ms)', title='(a) Operation medians')
    records = []
    for n in sizes:
        source = e1 if n == 5936 else e3
        record = next(x for x in source['targets'] if x['seed']==42 and x['regularization']==.001 and x['n']==n)
        records.append(record)
    for method in METHODS:
        y = np.array([x['timing'][method]['median_seconds'] for x in records])*1000
        quartiles = np.array([np.percentile([b['seconds'] for b in x['timing'][method]['blocks'] if b['all_converged']], [25,75], method='linear') for x in records])*1000
        color, ls = STYLES[method]
        axes[1].errorbar(sizes, y, yerr=[y-quartiles[:,0],quartiles[:,1]-y], marker='o', label=LABELS[method], color=color, ls=ls, capsize=3)
    axes[1].set(xscale='log', yscale='log', xlabel='Training triples N', ylabel='Solve time (ms)', title='(b) Solve medians')
    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()
    fig.legend(handles0 + handles1, labels0 + labels1, loc='outside upper center', ncol=3, frameon=False)
    for ax in axes:
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_xticks([], minor=True)
    data['size_microbenchmark_seconds'] = micro
    data['size_solve_source'] = {'512': 'E3', '2048': 'E3', '5936': 'E1, same seed and lambda'}
    finish(fig, out, 'fig3-sample-size')


def retrieval_data(validation):
    baseline = next(x for x in validation['systems'] if x['regularization'] is None)
    records = []
    for lam in LAMBDAS:
        for metric in ['nDCG@10', 'Recall@100']:
            values = np.mean([x['metrics'][metric] for x in validation['systems'] if x['regularization'] == lam], axis=0)
            comp = next(x for x in validation['comparisons'] if x['regularization']==lam and x['metric']==metric)
            b = float(np.mean(baseline['metrics'][metric]))
            records.append({'regularization':lam, 'metric':metric, 'mean':float(values.mean()), 'baseline':b,
                'paired_difference':comp['mean'], 'difference_ci95':comp['ci95'],
                'baseline_shifted_difference_ci95':[b+x for x in comp['ci95']]})
    return records


def fig4(validation, test, posthoc, out, data):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8), layout='constrained')
    records = retrieval_data(validation)
    for ax, metric, title in [(axes[0,0], 'nDCG@10', '(a) Validation nDCG@10'), (axes[0,1], 'Recall@100', '(b) Validation Recall@100')]:
        rows = [x for x in records if x['metric']==metric]
        x = [r['regularization'] for r in rows]
        low = [r['baseline_shifted_difference_ci95'][0] for r in rows];high=[r['baseline_shifted_difference_ci95'][1] for r in rows]
        ax.fill_between(x, low, high, color=PALETTE['blue'], alpha=.18, linewidth=0, label='Baseline + paired 95% CI')
        ax.plot(x, [r['mean'] for r in rows], 'o-', color=PALETTE['blue'], label='Mean across seeds')
        ax.axhline(rows[0]['baseline'], ls='--', color='0.4', label='Identity baseline')
        ax.set(xscale='log', xlabel=r'$\lambda$', ylabel=metric, title=title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    rows = [next(x for x in posthoc['systems'] if x['seed']==42 and x['regularization']==lam) for lam in LAMBDAS]
    ax = axes[1,0]
    ax.plot(LAMBDAS, [x['train_metrics']['nDCG@10'] for x in rows], 'o-', color=PALETTE['blue'], label='nDCG@10')
    ax.axhline(posthoc['baseline']['train_metrics']['nDCG@10'], color='0.5', ls='--', label='Identity nDCG@10')
    ax.plot(LAMBDAS, [x['hard_pair_accuracy'] for x in rows], 's:', color=PALETTE['vermillion'], label='Hard-pair acc.')
    ax.set(xscale='log', xlabel=r'$\lambda$', ylabel='Metric value', title='(c) Post-hoc training, seed 42')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, -.38), ncol=2, columnspacing=.8, handlelength=1.5, labelspacing=.3, borderaxespad=0, frameon=False)
    ax = axes[1,1]
    comps = test['comparisons']
    for i, row in enumerate(comps):
        lo,hi=row['ci95'];ax.plot([lo,hi],[i,i], color=PALETTE['blue'], lw=2)
        ax.plot(row['mean'],i,'o',color=PALETTE['blue'])
    ax.axvline(0,color='0.6',ls='--')
    ax.set_yticks(range(len(comps)),[x['metric'] for x in comps])
    ax.set(xlabel='Paired difference vs identity', title=r'(d) Secondary test, $\lambda=10^{-3}$')
    ax.margins(y=.5)
    data['validation'] = records
    data['posthoc_figure_scope'] = 'seed 42 only, frozen parameters; reconstructed binary training relevance; descriptive post-hoc'
    data['validation_interval_semantics'] = 'baseline mean + paired-difference CI, not an independent confidence interval for the adapted mean'
    finish(fig, out, 'fig4-retrieval')


def csv_write(path, fields, rows):
    with path.open('w', newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(rows)


def tables(groups, validation, test, tables_out):
    rows=[]
    for x in groups:
        rows.append({'lambda':x['regularization'],'method':LABELS[x['method']], 'iteration_min':x['iterations_range'][0], 'iteration_max':x['iterations_range'][1],
            'median_of_seed_medians_ms':x['median_of_seed_medians_seconds']*1000,
            'within_seed_iqr_min_ms':x['within_seed_iqr_range_seconds'][0]*1000,
            'within_seed_iqr_max_ms':x['within_seed_iqr_range_seconds'][1]*1000,'successful_seeds':x['successes']})
    csv_write(tables_out/'table1-e1.csv',list(rows[0]),rows)
    lines=[r'\begin{table}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\caption{E1 convergence and independent timing. Updates span three seeds; time is the median of their timing medians. The IQR range spans three within-seed IQRs, each from five blocks, with method-specific block sizes. All 36 configurations converged. Success counts converged seeds out of three.}',r'\label{tab:e1}',r'\begin{tabular}{llrrrr}',r'\toprule',r'$\lambda$ & Method & Updates & Time (ms) & IQR range (ms) & Success \\',r'\midrule']
    md=['| Lambda | Method | Updates min-max | Median of seed medians (ms) | Within-seed IQR range (ms) | Success |','| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        it=str(row['iteration_min']) if row['iteration_min']==row['iteration_max'] else f"{row['iteration_min']}-{row['iteration_max']}";iq=f"{row['within_seed_iqr_min_ms']:.3f}-{row['within_seed_iqr_max_ms']:.3f}"
        lines.append(f"{ltex(row['lambda'])} & {row['method']} & {it} & {row['median_of_seed_medians_ms']:.3f} & {iq} & {row['successful_seeds']}/3 \\")
        md.append(f"| {row['lambda']:g} | {row['method']} | {it} | {row['median_of_seed_medians_ms']:.3f} | {iq} | {row['successful_seeds']}/3 |")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    lines=[line.rstrip(chr(92)) + chr(92)*2 if ' & ' in line else line for line in lines]
    (tables_out/'table1-e1.tex').write_text('\n'.join(lines)+'\n')
    (tables_out/'table1-e1.md').write_text('\n'.join(md)+'\n\nIQR range is across within-seed IQRs, not a pooled IQR. Raw values are in the CSV.\n')
    records=retrieval_data(validation);validation_rows=[]
    for lam in LAMBDAS:
        row={'split':'validation','lambda':lam,'seed':'mean of 42/43/44','identity':'not selected primary'}
        for metric,short in [('nDCG@10','ndcg'),('Recall@100','recall')]:
            x=next(x for x in records if x['regularization']==lam and x['metric']==metric)
            row.update({short+'_mean':x['mean'],short+'_difference':x['paired_difference'],short+'_ci_low':x['difference_ci95'][0],short+'_ci_high':x['difference_ci95'][1]})
        validation_rows.append(row)
    baseline=next(x for x in validation['systems'] if x['regularization'] is None)
    baseline_row={'split':'validation','lambda':'identity','seed':'-','identity':'primary'}
    for metric,short in [('nDCG@10','ndcg'),('Recall@100','recall')]:baseline_row.update({short+'_mean':float(np.mean(baseline['metrics'][metric])),short+'_difference':0.,short+'_ci_low':0.,short+'_ci_high':0.})
    validation_rows.insert(0,baseline_row)
    test_rows=[{'split':'test','lambda':x['regularization'] if x['regularization'] is not None else 'identity','seed':x['seed'] if x['seed'] is not None else '-', 'identity':'primary' if x['regularization'] is None else 'secondary','ndcg_mean':x['aggregate']['nDCG@10'],'recall_mean':x['aggregate']['Recall@100']} for x in test['per_seed']]
    test_mean={'split':'test','lambda':.001,'seed':'mean of 42/43/44','identity':'secondary'}
    for metric,short in [('nDCG@10','ndcg'),('Recall@100','recall')]:
        comp=next(x for x in test['comparisons'] if x['metric']==metric)
        test_mean.update({short+'_mean':float(np.mean([x[short+'_mean'] for x in test_rows if x['lambda']!='identity'])),short+'_difference':comp['mean'],short+'_ci_low':comp['ci95'][0],short+'_ci_high':comp['ci95'][1]})
    test_rows.append(test_mean)
    fields=list(dict.fromkeys(k for x in validation_rows+test_rows for k in x))
    csv_write(tables_out/'table2-retrieval.csv',fields,validation_rows+test_rows)
    lines=[r'\begin{table}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\caption{Retrieval results. Validation metrics average seeds within queries; 95\% intervals describe paired differences from identity. Identity was selected as primary by the pre-specified rule; ref. denotes self-comparison. The $10^{-3}$ comparison remains secondary on test. Its mean row averages seeds within queries before paired inference; individual seed rows are descriptive.}',r'\label{tab:retrieval}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Validation & nDCG@10 & $\Delta$ 95\% CI & Recall@100 & $\Delta$ 95\% CI \\',r'\midrule']
    md=['| Validation system | nDCG@10 | Paired difference CI | Recall@100 | Paired difference CI |','| --- | --- | --- | --- | --- |']
    for x in validation_rows:
        name='Identity (primary)' if x['lambda']=='identity' else ltex(x['lambda'])
        a=f"[{x['ndcg_ci_low']:.4f}, {x['ndcg_ci_high']:.4f}]";b=f"[{x['recall_ci_low']:.4f}, {x['recall_ci_high']:.4f}]"
        if x['lambda']=='identity':a=b='ref.'
        lines.append(f"{name} & {x['ndcg_mean']:.4f} & {a} & {x['recall_mean']:.4f} & {b} \\")
        md.append(f"| {name} | {x['ndcg_mean']:.4f} | {a} | {x['recall_mean']:.4f} | {b} |")
    lines += [r'\midrule',r'Test system & nDCG@10 & $\Delta$ 95\% CI & Recall@100 & $\Delta$ 95\% CI \\',r'\midrule']
    md += ['', '| Test system | nDCG@10 | Paired difference CI | Recall@100 | Paired difference CI |', '| --- | --- | --- | --- | --- |']
    for x in test_rows:
        if x['lambda']=='identity':
            name='Identity (primary)';a=b='ref.'
        elif x['seed']=='mean of 42/43/44':
            name=r'$10^{-3}$, seed mean'
            a=f"[{x['ndcg_ci_low']:.4f}, {x['ndcg_ci_high']:.4f}]";b=f"[{x['recall_ci_low']:.4f}, {x['recall_ci_high']:.4f}]"
        else:
            name=r'$10^{-3}$, seed '+str(x['seed']);a=b=''
        lines.append(f"{name} & {x['ndcg_mean']:.4f} & {a} & {x['recall_mean']:.4f} & {b} " + chr(92)*2)
        md.append(f"| {name} | {x['ndcg_mean']:.4f} | {a} | {x['recall_mean']:.4f} | {b} |")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    lines=[line.rstrip(chr(92)) + chr(92)*2 if ' & ' in line else line for line in lines]
    (tables_out/'table2-retrieval.tex').write_text('\n'.join(lines)+'\n')
    (tables_out/'table2-retrieval.md').write_text('\n'.join(md)+'\n\nref. denotes self-comparison. The secondary test mean averages three seeds within each query; its confidence interval concerns the paired difference from identity.\n')


def review_html(figures, tables_out, posthoc, test):
    import html
    def md_table(path):
        result=[];opened=False
        for line in path.read_text().splitlines():
            if line.startswith('|'):
                cells=[x.strip() for x in line.strip('|').split('|')]
                if all(set(x)<=set('-: ') for x in cells):continue
                if not opened:result.append('<table>');opened=True
                result.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
            else:
                if opened:result.append('</table>');opened=False
                if line:result.append('<p>'+html.escape(line)+'</p>')
        if opened:result.append('</table>')
        return ''.join(result)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>COMP6704 figures review</title>',
        '<style>body{font:16px system-ui;margin:32px auto;max-width:1280px;padding:0 20px;color:#171717}img{width:100%;height:auto;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}td{padding:7px;border-bottom:1px solid #ddd}tr:first-child{font-weight:bold;background:#eee}.note{background:#f3f3f3;padding:15px;line-height:1.6}h2{margin-top:40px}</style>',
        '<h1>Figures and tables for review</h1><p class="note">Figures read saved results; post-hoc training metrics use binary relevance reconstructed from training triples. The frozen primary is identity; the adapter test comparison is secondary. No new validation/test retrieval or optimization.</p>']
    for i,stem in enumerate(['fig1-convergence','fig2-regularization-cost','fig3-sample-size','fig4-retrieval'],1):
        parts.append(f'<h2>Figure {i}</h2><a href="figures/{stem}.pdf">Vector PDF</a><img src="figures/{stem}.png" alt="Figure {i}">')
    for name,title in [('table1-e1','E1 timing table'),('table2-retrieval','Retrieval table')]:
        parts.append('<h2>'+title+'</h2>'+md_table(tables_out/(name+'.md')))
        parts.append(f'<a href="tables/{name}.tex">LaTeX</a> | <a href="tables/{name}.csv">CSV</a>')
    parts.append('<h2>Post-hoc training statistics: identity + 12 systems</h2><p class="note">'+html.escape(posthoc['qrels_definition'])+'</p><table><tr><td>Lambda</td><td>Seed</td><td>Parameter distance</td><td>Negative dimensions</td><td>Hard-pair accuracy</td><td>Binary train nDCG@10</td><td>Train Recall@100</td></tr>')
    for row in [posthoc['baseline']]+posthoc['systems']:
        vals=[str(row.get('regularization') or 'identity'),str(row.get('seed') or '-'),f"{row['parameter_distance']:.6f}",str(row['negative_dimensions']),f"{row['hard_pair_accuracy']:.6f}",f"{row['train_metrics']['nDCG@10']:.6f}",f"{row['train_metrics']['Recall@100']:.6f}"]
        parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in vals)+'</tr>')
    parts.append('</table><h2>Secondary test paired differences</h2><table><tr><td>Metric</td><td>Mean difference</td><td>95% CI</td></tr>')
    for row in test['comparisons']:
        parts.append(f"<tr><td>{html.escape(row['metric'])}</td><td>{row['mean']:.9f}</td><td>[{row['ci95'][0]:.9f}, {row['ci95'][1]:.9f}]</td></tr>")
    parts.append('</table><p>Timing: figure 2 uses median of seed medians and seed min to max ranges. Table 1 reports within-seed IQR ranges, not a pooled IQR. Figure 3 uses seed 42 medians and actual Q1 to Q3 blocks; full N reuses E1. Figure 2 cost components are assumption-based proxies, not measured wall-time attribution. Figure 4 validation bands shift paired-difference intervals by baseline; they are not independent adapted-mean intervals.</p></html>')
    (figures.parent/'review.html').write_text(''.join(parts))


def validate_outputs(figures, tables_out, replace):
    # Only known generated files may be replaced; anything else in the output stops the rebuild.
    for output in (figures,tables_out):
        if output.exists() and any(output.iterdir()) and not replace:raise FileExistsError('Report output is nonempty; --replace-generated is required to rebuild owned artifacts')
    if figures.parent.joinpath('review.html').exists() and not replace:raise FileExistsError('Existing review.html requires --replace-generated')
    allowed_figs={f'{stem}.{ext}' for stem in ['fig1-convergence','fig2-regularization-cost','fig3-sample-size','fig4-retrieval'] for ext in ['pdf','png']}|{'figure-data.json'}
    allowed_tables={f'{stem}.{ext}' for stem in ['table1-e1','table2-retrieval'] for ext in ['tex','csv','md']}
    for output,names in [(figures,allowed_figs),(tables_out,allowed_tables)]:
        if output.exists() and any(p.name not in names for p in output.iterdir()):raise ValueError('Unknown report artifact found; refusing replacement')



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/runs/full-experiment-001')
    p.add_argument('--calibration',type=Path,default=ROOT/'artifacts/runs/adam-rates')
    p.add_argument('--posthoc',type=Path,default=ROOT/'artifacts/analysis/posthoc-001/summary.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/report',help='Directory that receives figures/, tables/, and review.html')
    p.add_argument('--replace-generated',action='store_true',help='Explicitly rebuild only known generated report files')
    args=p.parse_args();args.figures=args.output/'figures';args.tables=args.output/'tables'
    validate_outputs(args.figures,args.tables,args.replace_generated);configure()
    e1=read(args.run/'e1-summary.json');e3=read(args.run/'e3-summary.json')
    validation=read(args.run/'e2/validation.json');test=read(args.run/'e2/summary.json');posthoc=read(args.posthoc)
    if e1['mode']!='formal' or e3['mode']!='formal':raise ValueError('Only saved formal results are report inputs')
    args.figures.mkdir(parents=True,exist_ok=True);args.tables.mkdir(parents=True,exist_ok=True)
    groups=time_groups(e1)
    data={'source_run':str(args.run),'calibration_run':str(args.calibration),'posthoc':str(args.posthoc),
        'scope':'read-only saved-result plotting; no optimization, retrieval, qrels access or mutation of run directories',
        'timing_aggregation':'median of three seed-specific medians; figure bars use min-max seed medians; table IQR range is min-max within-seed IQRs, not pooled',
        'time_groups':groups}
    fig1(args.run,args.calibration,args.figures);fig2(args.run,groups,args.figures,data)
    fig3(args.run,e1,e3,args.figures,data);fig4(validation,test,posthoc,args.figures,data)
    tables(groups,validation,test,args.tables);write(args.figures/'figure-data.json',data)
    review_html(args.figures,args.tables,posthoc,test)
    print(json.dumps({'pdfs':4,'png_previews':4,'latex_tables':2,'output':str(args.figures)}))


if __name__=='__main__':main()
