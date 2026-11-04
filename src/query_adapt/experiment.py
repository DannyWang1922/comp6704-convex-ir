"""Bounded training-only spectrum diagnostics and seed 42 calibration."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize as scipy_minimize
from threadpoolctl import threadpool_limits
from .data import read_json, write_json, make_pairs
from .prepare import load_embeddings
from .objective import PairwiseObjective
from .solvers import minimize, SolverConfig
from .pilot import Tee, update_status


def spectrum(x, a, regularization=0.0):
    z = x @ a
    weights = expit(z) * expit(-z)
    h = (x.T * weights) @ x / len(x)
    values = np.linalg.eigvalsh(h)
    threshold = float(max(h.shape) * np.finfo(float).eps * max(values[-1], 0))
    active = values[values > threshold]
    singular = np.linalg.svd(x, compute_uv=False)
    rank_threshold = float(max(x.shape) * np.finfo(float).eps * singular[0])
    return {'data_eigenvalues': values.tolist(), 'effective_threshold': threshold,
            'effective_dimension': len(active), 'effective_data_condition': float(active[-1]/active[0]) if len(active) else None,
            'full_condition': float((values[-1]+regularization)/(max(0, values[0])+regularization)) if regularization else None,
            'rank': int(np.sum(singular > rank_threshold)), 'rank_threshold': rank_threshold,
            'global_curvature_bound': float(singular[0]**2/(4*len(x)))}


def groups(x, rows, a):
    z = x @ a
    weight = expit(-z)
    result, vectors = {}, {}
    for kind in ('hard', 'random'):
        mask = np.array([r['kind'] == kind for r in rows])
        vector = -(x[mask].T @ weight[mask])/len(x)
        vectors[kind] = vector
        result[kind] = {'count': int(mask.sum()), 'mean_margin': float(z[mask].mean()),
                        'negative_margin_fraction': float(np.mean(z[mask] < 0)),
                        'scalar_weight_sum': float(weight[mask].sum()),
                        'gradient_norm': float(np.linalg.norm(vector)), 'gradient_vector': vector.tolist()}
    denominator = np.linalg.norm(vectors['hard'])*np.linalg.norm(vectors['random'])
    cosine = float(np.clip(vectors['hard'] @ vectors['random']/denominator, -1, 1)) if denominator else None
    result['gradient_cosine'] = cosine
    result['gradient_angle_degrees'] = float(np.degrees(np.arccos(cosine))) if cosine is not None else None
    return result


def select_adam(results):
    successful = [(rate, result) for rate, result in results if result.status == 'converged']
    return min(successful, key=lambda item: (item[1].history[-1]['iteration'], item[0]))[0] if successful else None



def next_adam_rate(results, policy="legacy"):
    """Predeclared boundary extensions; only successful candidates can trigger."""
    if policy not in {"legacy", "upper03"}:
        raise ValueError("unknown Adam extension policy")
    rates = {rate for rate, _ in results}
    if not {.003, .01, .03} <= rates:
        raise ValueError("initial Adam candidate grid is incomplete")
    initial_best = select_adam([(r, result) for r, result in results if r in {.003, .01, .03}])
    first = .1 if initial_best == .03 else .001 if initial_best == .003 else None
    if first is not None and first not in rates:
        return first
    if policy == "upper03" and select_adam(results) == .1 and .3 not in rates:
        return .3
    return None


def milestones(history):
    result = {}
    for bound in (1e-5, 1e-7, 1e-9):
        result[f'B<={bound:g}'] = next(({k: row[k] for k in ('iteration', 'seconds')} for row in history if row['certificate_B'] <= bound), None)
    result['gradient<=1e-6'] = next(({k: row[k] for k in ('iteration', 'seconds')} for row in history if row['gradient_norm'] <= 1e-6), None)
    return result


def initialize(config_path, output):
    config = tomllib.loads(config_path.read_text())
    root = config_path.parent.parent
    setup = tomllib.loads((root / config['setup_config']).read_text())
    if config.get('require_clean_tracked') and subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--','src/query_adapt','tools','configs','tests'],cwd=root,text=True).strip():
        raise RuntimeError('calibration requires tracked and committed source, tools, configs and tests')
    if not output.exists():
        output.mkdir(parents=True)
        write_json(output/'manifest.json', {'config': config, 'setup': setup,
                   'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                   'git_diff': subprocess.check_output(['git','diff'], cwd=root, text=True),
                   'diagnostics_source_run': config.get('diagnostics_source_run'), 'adam_policy': config.get('adam_policy','recalibrate'), 'scope': 'three training diagnostics, seed42 calibration only', 'validation_evaluation': False, 'test_evaluation': False})
        shutil.copytree(Path(__file__).parent, output/'source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(config_path, output/'calibration.toml')
        shutil.copy2(root/config['setup_config'], output/'setup.toml')
    else:
        saved = read_json(output/'manifest.json')
        if saved['config'] != config or saved['setup'] != setup:
            raise ValueError('resume configuration mismatch')
        source = output/'source'
        current = Path(__file__).parent
        if set(p.name for p in current.glob('*.py')) != set(p.name for p in source.glob('*.py')) or any(p.read_bytes() != (source/p.name).read_bytes() for p in current.glob('*.py')):
            raise ValueError('resume source differs from immutable run snapshot')
    return config, root, setup


def diagnostics(config, root, setup, output):
    if config.get('diagnostics_source_run'):
        source=(root/config['diagnostics_source_run']).resolve()
        if read_json(source/'manifest.json')['setup']!=setup:
            raise ValueError('source diagnostics setup differs from current setup')
        frozen=read_json(source/'freeze.json')
        if frozen['lambdas']!=config['lambdas'] or frozen['seeds']!=config['diagnostic_seeds']:
            raise ValueError('source diagnostics freeze mismatch')
        target=output/'diagnostics';target.mkdir(exist_ok=True)
        for seed in config['diagnostic_seeds']:
            metadata=read_json(source/'diagnostics'/f'seed{seed}.json')
            if metadata['seed']!=seed or metadata['pairs']!=dict(setup['pairs'],seed=seed):
                raise ValueError('source pair configuration mismatch')
            for suffix in ('npz','json'):
                original=source/'diagnostics'/f'seed{seed}.{suffix}';destination=target/original.name
                if destination.exists():
                    if destination.read_bytes()!=original.read_bytes():raise ValueError('copied diagnostics differ')
                else:shutil.copy2(original,destination)
            print(f'DIAGNOSTICS reused seed={seed} source={source}',flush=True)
        receipt={'source_run':str(source),'source_manifest':str(source/'manifest.json'),'source_commit':read_json(source/'manifest.json')['git_commit'],'copied_training_caches':True,'new_spectrum_computation':False}
        if not (output/'diagnostics-provenance.json').exists():write_json(output/'diagnostics-provenance.json',receipt)
        return
    docs, queries, dids, qids, splits, qrels = load_embeddings(setup, root)
    target = output/'diagnostics'
    target.mkdir(exist_ok=True)
    for seed in config['diagnostic_seeds']:
        path = target/f'seed{seed}.json'
        if path.exists():
            print(f'DIAGNOSTICS seed={seed} already complete; reusing', flush=True)
            continue
        pair_config = dict(setup['pairs'], seed=seed)
        x, rows = make_pairs(docs, queries, dids, qids, qrels, splits['train'], **pair_config)
        cache = target/f'seed{seed}.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as old:
                if not np.array_equal(old['features'], x):
                    raise ValueError('partial cache does not reproduce')
        else:
            with cache.open('xb') as stream:
                np.savez_compressed(stream, features=x)
        info = {'seed': seed, 'pairs': pair_config, 'shape': list(x.shape), 'triples': rows,
                'spectrum': spectrum(x, np.ones(x.shape[1])), 'groups': groups(x, rows, np.ones(x.shape[1]))}
        write_json(path, info)
        print(f'DIAGNOSTICS seed={seed} N={len(x)} rho={info["spectrum"]["data_eigenvalues"][-1]:.9g} effective_dim={info["spectrum"]["effective_dimension"]}', flush=True)
    print('DIAGNOSTICS COMPLETE. Freeze must be recorded before calibration.', flush=True)


def freeze(config, output):
    records = [read_json(output/'diagnostics'/f'seed{seed}.json') for seed in config['diagnostic_seeds']]
    rho = [r['spectrum']['data_eigenvalues'][-1] for r in records]
    if max(rho)/min(rho) > 2:
        raise ValueError('seed curvature differs by more than factor 2; review before freezing')
    write_json(output/'freeze.json', {'lambdas': config['lambdas'], 'seeds': config['diagnostic_seeds'],
               'rho': rho, 'reason': 'training spectrum same order across seeds; predeclared grid retained',
               'adam_selection': 'successful candidates: fewest updates, tie smaller rate; predeclared boundary policy', 'adam_extension_policy':config.get('adam_extension_policy','legacy'),
               'validation_read': False, 'test_read': False})


def calibrate(config, output, root=None):
    if config.get('adam_policy')=='pending':
        raise RuntimeError('Adam policy decision pending; calibration is disabled')
    frozen = read_json(output/'freeze.json')
    if frozen['lambdas'] != config['lambdas']:
        raise ValueError('frozen grid mismatch')
    with np.load(output/'diagnostics/seed42.npz', allow_pickle=False) as cache:
        x = cache['features']
    rows = read_json(output/'diagnostics/seed42.json')['triples']
    active_file = output/'active-solve.json'
    if active_file.exists():
        raise RuntimeError('unfinished solve reservation: inspect active-solve.json and budget before recovery')
    budget_file = output/'budget.jsonl'
    used = sum(json.loads(line)['seconds'] for line in budget_file.read_text().splitlines()) if budget_file.exists() else 0.0
    prior_receipts=[]
    for previous in config.get('inherited_budget_runs',[]):
        path=((Path(root) if root is not None else output.parents[2])/previous/'budget.jsonl').resolve()
        # Calibration output normally lives in <root>/artifacts/runs/<id>.
        content=path.read_text()
        prior_receipts.append({'path':str(path),'content':content,'seconds':sum(json.loads(line)['seconds'] for line in content.splitlines())})
    inherited=sum(row['seconds'] for row in prior_receipts)
    receipt=output/'inherited-budget.json'
    if prior_receipts:
        if receipt.exists():
            if read_json(receipt)!=prior_receipts:raise ValueError('inherited budget sources differ')
        else:write_json(receipt,prior_receipts)
    summaries = []
    def charge(label, seconds):
        nonlocal used
        used += seconds
        with budget_file.open('a') as stream:
            stream.write(json.dumps({'label': label, 'seconds': seconds, 'cumulative_seconds': used})+'\n')
    for lam in frozen['lambdas']:
        objective = PairwiseObjective(x, lam)
        directory = output/f'lambda-{lam:g}'
        directory.mkdir(exist_ok=True)
        base = SolverConfig(**config['solver'])
        def solve(label, method, cfg=base, initial=None):
            path = directory/f'{label}.json'
            if path.exists():
                from .solvers import SolverResult
                saved = read_json(path)
                if saved["status"] in {"nonfinite", "nonfinite_hessian", "nonfinite_update", "invalid_direction", "linear_solve_failed"}:
                    raise ArithmeticError(f"saved numerical failure requires review: {label}: {saved['status']}")
                print(f'RESUME completed {label} lambda={lam:g}', flush=True)
                return SolverResult(saved['method'], np.array(saved['parameters']), saved['status'], saved['history'], saved['costs'], saved['settings'])
            remaining = config['total_solver_seconds'] - inherited - used
            if remaining <= 0:
                raise TimeoutError('total calibration solver budget exhausted')
            cfg = SolverConfig(**dict(asdict(cfg), max_seconds=min(cfg.max_seconds, remaining)))
            update_status(output, {'state': 'running', 'stage': 'seed42 calibration', 'lambda': lam, 'method': label, 'solver_seconds': used, 'inherited_solver_seconds':inherited, 'budget_seconds': config['total_solver_seconds'], 'metric': 'B=||g||^2/(2lambda), parameter bound=||g||/lambda; both required'})
            print(f'START seed=42 lambda={lam:g} method={label} remaining_budget={remaining:.2f}s', flush=True)
            write_json(active_file, {'label': f'{lam:g}/{label}', 'reserved_seconds': cfg.max_seconds})
            tick = time.perf_counter()
            try:
                result = minimize(objective, method, cfg, initial=initial)
            finally:
                charge(f'{lam:g}/{label}', time.perf_counter()-tick)
                active_file.unlink()
            result.save(path)
            final = result.history[-1]
            print(f'END {label} status={result.status} updates={final["iteration"]} B={final["certificate_B"]:.3e} g/lambda={final["parameter_error_bound"]:.3e} time={result.costs["total_seconds"]:.4f}s', flush=True)
            if result.status in {'nonfinite','nonfinite_hessian','nonfinite_update','invalid_direction','linear_solve_failed'}:
                raise ArithmeticError(f'numerical failure {label}: {result.status}')
            return result
        gd = solve('gd', 'gd')
        newton = solve('newton', 'newton')
        unit = solve('gd-unit', 'gd', SolverConfig(**dict(asdict(base), gd_step_mode='unit')))
        candidates = [(rate, solve(f'adam-{rate:g}', 'adam', SolverConfig(**dict(asdict(base), learning_rate=rate)))) for rate in config['adam_rates']]
        initial_best = select_adam(candidates)
        extensions = []
        while (extension := next_adam_rate(candidates, config.get('adam_extension_policy','legacy'))) is not None:
            extensions.append(extension)
            candidates.append((extension, solve(f'adam-{extension:g}', 'adam', SolverConfig(**dict(asdict(base), learning_rate=extension)))))
        extension = extensions[0] if extensions else None
        selected = select_adam(candidates)
        strict = SolverConfig(**dict(asdict(base), objective_certificate=1e-12, parameter_certificate=1e-5, max_seconds=config['reference_seconds']))
        refined = solve('reference-newton', 'newton', strict, newton.parameters)
        lbpath = directory/'reference-lbfgsb.json'
        if not lbpath.exists():
            remaining = min(config['reference_seconds'], config['total_solver_seconds']-inherited-used)
            if remaining <= 0:
                raise TimeoutError('total solver budget exhausted')
            write_json(active_file, {'label': f'{lam:g}/reference-lbfgsb', 'reserved_seconds': remaining})
            tick = time.perf_counter()
            counts = {'Xv': 0, 'XTv': 0}
            def fun(a):
                if time.perf_counter()-tick >= remaining:
                    raise TimeoutError('L-BFGS-B time limit')
                counts['Xv'] += 1; counts['XTv'] += 1
                return objective.value_gradient(a)
            try:
                lb = scipy_minimize(fun, np.ones(objective.dimension), jac=True, method='L-BFGS-B', options={'gtol': min(np.sqrt(2*lam*1e-12), lam*1e-5)/np.sqrt(objective.dimension), 'ftol': 1e-15, 'maxiter': 20000, 'maxls': 60})
                value, grad = fun(lb.x)
                norm = float(np.linalg.norm(grad))
                saved = {'success': bool(lb.success), 'message': str(lb.message), 'parameters': lb.x.tolist(), 'objective': value, 'gradient_norm': norm, 'certificate_B': norm**2/(2*lam), 'parameter_error_bound': norm/lam, 'counts': counts, 'iterations': int(lb.nit), 'seconds': time.perf_counter()-tick}
            except TimeoutError as exc:
                saved = {'success': False, 'message': str(exc), 'seconds': time.perf_counter()-tick}
            finally:
                charge(f'{lam:g}/reference-lbfgsb', time.perf_counter()-tick)
                active_file.unlink()
            write_json(lbpath, saved)
        lb = read_json(lbpath)
        refs = []
        if refined.status == 'converged':
            refs.append((refined.history[-1]['objective'], refined.history[-1]['certificate_B'], refined.parameters, 'newton'))
        if lb.get('certificate_B', float('inf')) <= 1e-12 and lb.get('parameter_error_bound', float('inf')) <= 1e-5:
            refs.append((lb['objective'], lb['certificate_B'], np.array(lb['parameters']), 'lbfgsb'))
        if not refs:
            raise ArithmeticError(f'no certified reference lambda={lam}')
        ref_value, ref_B, ref_a, ref_method = min(refs, key=lambda r:r[0])
        checks = []
        for label, result in [('gd',gd),('newton',newton),('gd-unit',unit)]+[(f'adam-{rate:g}',r) for rate,r in candidates]:
            final = result.history[-1]
            delta = 100*np.finfo(float).eps*max(1,abs(final['objective']),abs(ref_value))
            gap = final['objective']-ref_value
            checks.append({'method': label, 'status': result.status, 'iterations': final['iteration'], 'seconds': result.costs['total_seconds'], 'gap': gap, 'gap_bound': final['certificate_B']+ref_B+delta, 'gap_check': bool(abs(gap)<=final['certificate_B']+ref_B+delta), 'milestones': milestones(result.history)})
        summary = {'lambda': lam, 'selected_adam_rate': selected, 'initial_best': initial_best, 'extension': extension, 'extensions': extensions, 'reference_method': ref_method, 'reference_B': ref_B, 'reference_objective': ref_value, 'reference_spectrum': spectrum(x,ref_a,lam), 'reference_groups':groups(x,rows,ref_a), 'checks':checks, 'reference_parameter_distance': float(np.linalg.norm(refined.parameters-np.array(lb['parameters']))) if 'parameters' in lb else None}
        if not (directory/'summary.json').exists():
            write_json(directory/'summary.json',summary)
        summaries.append(summary)
    state = 'complete' if all(r['status']=='converged' and r['gap_check'] for s in summaries for r in s['checks']) else 'complete_with_failures'
    final = {'state':state,'scope':'seed42 calibration only','solver_seconds':used,'inherited_solver_seconds':inherited,'cumulative_solver_seconds':used+inherited,'results':summaries,'validation_evaluation':False,'test_evaluation':False,'formal_timing':False,'seed43_44_solves':False,'E3':False}
    if not (output/'summary.json').exists():
        write_json(output/'summary.json',final)
    update_status(output,final)
    print(f'CALIBRATION {state} total_solver_seconds={used:.3f}; STOP: no further stages authorized',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/adam-rates.toml'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage', choices=['diagnostics','freeze','calibrate'],required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    config,root,setup=initialize(args.config.resolve(),output)
    with (output/'console.log').open('a') as log, redirect_stdout(Tee(sys.stdout,log)),threadpool_limits(limits=config['threads']):
        try:
            if args.stage=='diagnostics': diagnostics(config,root,setup,output)
            elif args.stage=='freeze': freeze(config,output)
            else: calibrate(config,output,root)
        except BaseException as exc:
            update_status(output,{'state':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','error':str(exc),'stage':args.stage})
            with (output/'errors.log').open('a') as stream: stream.write(traceback.format_exc())
            raise

if __name__=='__main__': main()
