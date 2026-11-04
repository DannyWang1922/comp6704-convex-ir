"""Explicit E0 runner with live terminal output; never evaluates the test set."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
import numpy as np
from scipy.optimize import minimize as reference_minimize
from threadpoolctl import threadpool_limits
from .data import read_json, write_json
from .objective import PairwiseObjective
from .prepare import load_embeddings
from .retrieval import retrieve, evaluate
from .solvers import SolverConfig, minimize


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.flush()
    def flush(self):
        self.stream.flush()
        self.log.flush()


def update_status(output, value):
    temporary = output / 'progress.tmp'
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(output / 'progress.json')


def monitor(output):
    """Read-only task panel; leaves the final summary visible when finished."""
    previous = None
    while True:
        path = output / 'progress.json'
        state = read_json(path) if path.exists() else {'state': 'waiting for E0', 'completed': 0, 'total': 3}
        text = json.dumps(state, indent=2)
        if text != previous:
            print('\033[2J\033[HConvexIR E0 | Task Monitor\n', flush=True)
            print(f"State: {state['state']}\nCompleted: {state['completed']}/{state['total']}\n", flush=True)
            if 'method' in state:
                print(f"Current: {state['method']}  ({state.get('position', '-')}/{state['total']})", flush=True)
            if 'iteration' in state:
                print(f"Iteration: {state['iteration']}\nObjective: {state['objective']:.9f}\n"
                      f"Gradient: {state['gradient_norm']:.3e}\nSeconds: {state['seconds']:.2f}", flush=True)
            for row in state.get('results', []):
                print(f"{row['method']:8} {row['status']:16} iter={row['iterations']:4} time={row['seconds']:.3f}s", flush=True)
            print(f'\nResults: {output}\nTest evaluation: disabled', flush=True)
            previous = text
        if state['state'] in {'complete', 'complete_with_failures', 'failed', 'interrupted'}:
            return
        time.sleep(0.25)


def run(config_path, output):
    with config_path.open('rb') as stream:
        config = tomllib.load(stream)
    root = config_path.parent.parent
    with (root / config['setup_config']).open('rb') as stream:
        setup = tomllib.load(stream)
    output.mkdir(parents=True, exist_ok=False)
    summary = []
    state = {'state': 'preparing', 'completed': 0, 'total': len(config['methods']), 'results': summary}
    update_status(output, state)
    try:
        with (output / 'console.log').open('x') as log, redirect_stdout(Tee(sys.stdout, log)), threadpool_limits(limits=config['threads']):
            print('ConvexIR E0 | Initial pilot | No hyperparameter search', flush=True)
            print(f'Output: {output}\nConda Python: {sys.executable}', flush=True)
            docs, queries, dids, qids, splits, qrels = load_embeddings(setup, root)
            artifacts = root / setup['paths']['artifacts']
            pair_info = read_json(artifacts / 'train-pairs.json')
            if pair_info['config'] != setup['pairs']:
                raise ValueError('cached pair settings do not match setup')
            if any(row['query_id'] not in set(splits['train']) for row in pair_info['triples']):
                raise ValueError('non-training query in pair cache')
            with np.load(artifacts / 'train-pairs.npz', allow_pickle=False) as cache:
                all_features = cache['features']
            if len(all_features) != len(pair_info['triples']):
                raise ValueError('pair feature/ID count mismatch')
            selected = np.sort(np.random.default_rng(config['subset_seed']).choice(
                len(all_features), min(config['max_pairs'], len(all_features)), replace=False))
            objective = PairwiseObjective(all_features[selected], config['regularization'])
            solver_config = SolverConfig(**config['solver'])
            write_json(output / 'manifest.json', {'pilot': config, 'setup': setup,
                       'selected_pair_indices': selected.tolist(), 'feature_shape': list(objective.features.shape),
                       'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                       'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                       'test_evaluation': False, 'timing_note': 'Pilot wall times include live callback logging; not a benchmark.'})
            shutil.copytree(Path(__file__).parent, output / 'source', ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copy2(config_path, output / 'pilot.toml')
            print(f'Pairs: {len(selected)} | Dimension: {objective.dimension} | lambda: {config["regularization"]} | BLAS threads: {config["threads"]}', flush=True)
            baseline = read_json(artifacts / 'baseline-validation.json')['metrics']
            print(f'Validation baseline: nDCG@10={baseline["nDCG@10"]:.5f} Recall@100={baseline["Recall@100"]:.5f}', flush=True)
            for position, method in enumerate(config['methods'], 1):
                state.update(state='running', method=method, position=position)
                print(f'\n[{position}/{len(config["methods"])}] {method.upper()} START', flush=True)
                def on_iteration(row):
                    if row['iteration'] % config['log_every'] == 0 or row['gradient_norm'] <= solver_config.gradient_tolerance:
                        state.update(row)
                        update_status(output, state)
                        print(f"{method:6} iter={row['iteration']:4} F={row['objective']:.9f} "
                              f"|g|={row['gradient_norm']:.3e} elapsed={row['seconds']:.3f}s", flush=True)
                result = minimize(objective, method, solver_config, callback=on_iteration)
                result.save(output / f'{method}.json')
                final = result.history[-1]
                ranking = retrieve(docs, queries, dids, qids, splits['validation'], result.parameters)
                metrics = evaluate({q: qrels[q] for q in splits['validation']}, ranking)
                row = {'method': method, 'status': result.status, 'iterations': final['iteration'],
                       'objective': final['objective'], 'gradient_norm': final['gradient_norm'],
                       'seconds': result.costs['total_seconds'], 'validation': metrics}
                summary.append(row)
                state.update(completed=position, results=summary)
                update_status(output, state)
                print(f"{method.upper()} END status={result.status} time={row['seconds']:.3f}s "
                      f"nDCG@10={metrics['nDCG@10']:.5f} Recall@100={metrics['Recall@100']:.5f}", flush=True)
            state['state'] = 'reference_check'
            update_status(output, state)
            print('\nReference check: SciPy L-BFGS-B (same fixed objective)', flush=True)
            reference = reference_minimize(objective.value_gradient, np.ones(objective.dimension), jac=True,
                                           method='L-BFGS-B', options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 1000})
            ref_value, ref_grad = objective.value_gradient(reference.x)
            ref = {'success': bool(reference.success), 'message': str(reference.message),
                   'objective': ref_value, 'gradient_norm': float(np.linalg.norm(ref_grad))}
            write_json(output / 'reference.json', ref)
            for row in summary:
                row['objective_minus_reference'] = row['objective'] - ref_value
            passed = all(r['status'] == 'converged' and abs(r['objective_minus_reference']) < 1e-8 for r in summary)
            passed = passed and ref['success'] and ref['gradient_norm'] < 1e-6
            state.update(state='complete' if passed else 'complete_with_failures', reference=ref)
            write_json(output / 'summary.json', {'state': state['state'], 'baseline_validation': baseline,
                                                 'results': summary, 'reference': ref, 'test_evaluation': False})
            update_status(output, state)
            print('\nE0 SUMMARY', flush=True)
            for row in summary:
                print(f"{row['method']:8} {row['status']:16} iter={row['iterations']:4} "
                      f"time={row['seconds']:.3f}s gap={row['objective_minus_reference']:.3e}", flush=True)
            print(f"State: {state['state']} | Test evaluation: disabled\nSaved: {output}", flush=True)
    except BaseException as exc:
        state['state'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        state['error'] = str(exc)
        update_status(output, state)
        (output / 'error.txt').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/pilot.toml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    if args.monitor:
        monitor(args.output.resolve())
    else:
        run(args.config.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
