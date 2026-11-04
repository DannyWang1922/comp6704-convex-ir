"""Explicit E1/E3 orchestration. Default CLI action is read-only preflight."""
import argparse
from contextlib import redirect_stdout
import sys
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time
import tomllib
import numpy as np
from scipy.linalg import solve as linear_solve
from scipy.optimize import minimize as scipy_minimize
from threadpoolctl import threadpool_limits
from .data import read_json, write_json
from .objective import PairwiseObjective
from .solvers import SolverConfig, SolverResult, minimize
from .experiment import spectrum, groups, milestones, select_adam, next_adam_rate
from .pilot import update_status, Tee

METHODS = ('gd', 'newton', 'adam')
NUMERICAL = {'nonfinite', 'nonfinite_hessian', 'nonfinite_update', 'invalid_direction', 'linear_solve_failed', 'line_search_failed'}
CAPS = {'time_limit', 'max_iterations'}


def target_id(seed, lam, n):
    return f'seed{seed}-lambda{lam:g}-n{n}'


def load_config(path):
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text())
    for key in ('calibration_run','splits'):
        config[key] = str((path.parent.parent / config[key]).resolve())
    config['inherited_runs'] = [str((path.parent.parent/p).resolve()) for p in config.get('inherited_runs',[])]
    return config


def preflight(config):
    """Read cached training artifacts and certificates; no writes or solves."""
    run = Path(config['calibration_run'])
    frozen = read_json(run/'freeze.json')
    summary = read_json(run/'summary.json')
    frozen_config=read_json(run/'manifest.json')['config']
    if asdict(SolverConfig(**config['solver'])) != asdict(SolverConfig(**frozen_config['solver'])):
        raise ValueError('main solver configuration differs from frozen calibration')
    if frozen_config['solver'].get('gd_armijo')!=.5:
        raise ValueError('current campaign requires explicit v4 main-GD Armijo .5 calibration')
    if config.get('adam_extension_policy')!='upper03' or frozen_config.get('adam_extension_policy')!='upper03':
        raise ValueError('campaign and calibration must use the registered upper03 Adam policy')
    if config.get('threads')!=1 or config['reference_seconds']!=120 or not 0<config['e1_seconds']<=1800 or not 0<config['e3_seconds']<=900:
        raise ValueError('thread, reference or stage budgets differ from protocol')
    if frozen['lambdas'] != [1e-2, 1e-3, 1e-4, 1e-5] or frozen['seeds'] != [42,43,44]:
        raise ValueError('calibration grid differs from frozen protocol')
    for name in ('objective.py','solvers.py'):
        if (Path(__file__).parent/name).read_bytes() != (run/'source'/name).read_bytes():
            raise ValueError('numeric implementation differs from frozen calibration')
    rates = {}
    selected_candidates=[]
    old_gd=[]
    for row in summary['results']:
        rate = row['selected_adam_rate']
        if rate is None:
            raise ValueError('no successful frozen Adam rate')
        saved = read_json(run/f'lambda-{row["lambda"]:g}'/f'adam-{rate:g}.json')
        if saved['status'] != 'converged' or saved['history'][-1]['certificate_B'] > 1e-9 or saved['history'][-1]['parameter_error_bound'] > 1e-3:
            raise ValueError('selected calibration candidate is not certified')
        rates[str(row['lambda'])] = rate
        selected_candidates.append(str((run/f'lambda-{row["lambda"]:g}'/f'adam-{rate:g}.json').resolve()))
        old_path=run/f'lambda-{row["lambda"]:g}'/'gd-unit.json'
        old=read_json(old_path)
        old_gd.append({'regularization':row['lambda'],'path':str(old_path.resolve()),'status':old['status'],'final':old['history'][-1]})
    inherited=[]
    for previous in config.get('inherited_runs',[]):
        ledger=Path(previous)/'budget.jsonl'
        amount=sum(json.loads(line)['seconds'] for line in ledger.read_text().splitlines())
        inherited.append({'path':str(ledger.resolve()),'seconds':amount})
    caches = {}
    split_file = Path(config['splits'])
    splits = read_json(split_file)
    train_ids = set(splits['train'])
    for seed in frozen['seeds']:
        cache = run/'diagnostics'/f'seed{seed}.npz'
        metadata = run/'diagnostics'/f'seed{seed}.json'
        info = read_json(metadata)
        with np.load(cache, allow_pickle=False) as values:
            x = values['features']
        if x.shape != (5936,384) or not np.isfinite(x).all() or len(info['triples']) != len(x):
            raise ValueError('invalid training cache')
        if any(row['query_id'] not in train_ids for row in info['triples']):
            raise ValueError('cache includes non-training query')
        if info['seed'] != seed:
            raise ValueError('cache seed metadata mismatch')
        caches[str(seed)] = {'cache':str(cache.resolve()), 'metadata':str(metadata.resolve()), 'shape':list(x.shape)}
    return {'mode':'formal','frozen_lambdas':frozen['lambdas'],'planned_targets':protocol_module().planned_inventory()['main'][::3],'frozen_grid':frozen,'rates':rates,'caches':caches,'splits':str(split_file.resolve()),
            'old_gd_examples':old_gd,'selected_candidates':selected_candidates,'inherited_budgets':inherited,'freeze_file':str((run/'freeze.json').resolve()),'calibration_manifest':str((run/'manifest.json').resolve()), 'calibration_summary':str((run/'summary.json').resolve()),
            'new_retrieval_evaluation':False,'solver_executed':False}


def protocol_module():
    # Keep the independently tested inventory/timing policy in its existing file.
    import importlib.util
    path = Path(__file__).resolve().parents[2]/'tools/protocol.py'
    spec = importlib.util.spec_from_file_location('campaign_protocol',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(output, config, provenance, mode):
    output = Path(output)
    source = Path(__file__).parent
    tools = source.parents[1]/'tools'
    if not output.exists():
        output.mkdir(parents=True)
        write_json(output/'manifest.json',{'config':config,'provenance':provenance,'mode':mode,
                   'frozen_lambdas':provenance.get('frozen_lambdas',[.01]),'planned_targets':[{k:t[k] for k in ('seed','regularization','n')} for t in provenance.get('planned_targets',[])],
                   'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
                   'git_diff':subprocess.check_output(['git','diff'],cwd=source,text=True), 'created_time':time.time()})
        shutil.copytree(source,output/'source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(tools,output/'tools',ignore=shutil.ignore_patterns('__pycache__'))
        # Exact input snapshots avoid silent resume after calibration/cache changes.
        inputs = output/'inputs';inputs.mkdir()
        identity=[]
        for entry in provenance.get('caches',{}).values():
            for key in ('cache','metadata'):
                p=Path(entry[key]); destination=inputs/f'{len(identity)}-{p.name}'
                shutil.copy2(p,destination);identity.append({'original':str(p),'snapshot':destination.name})
        for item in provenance.get('old_gd_examples',[])+[{'path':p} for p in provenance.get('selected_candidates',[])]:
            p=Path(item['path']);destination=inputs/f'{len(identity)}-{p.name}'
            shutil.copy2(p,destination);identity.append({'original':str(p),'snapshot':destination.name})
        for entry in provenance.get('inherited_budgets',[]):
            p=Path(entry['path']);destination=inputs/f'{len(identity)}-{p.name}'
            shutil.copy2(p,destination);identity.append({'original':str(p),'snapshot':destination.name})
        for key in ('splits','calibration_manifest','calibration_summary','freeze_file'):
            if key in provenance:
                p=Path(provenance[key]);destination=inputs/f'{len(identity)}-{p.name}'
                shutil.copy2(p,destination);identity.append({'original':str(p),'snapshot':destination.name})
        write_json(output/'input-identity.json',identity)
    else:
        saved=read_json(output/'manifest.json')
        if saved['config'] != config or saved['provenance'] != provenance or saved['mode'] != mode:
            raise ValueError('campaign configuration/provenance mismatch')
        for live, old in ((source,output/'source'),(tools,output/'tools')):
            files={p.name for p in live.glob('*.py')}
            if files != {p.name for p in old.glob('*.py')} or any((live/name).read_bytes()!=(old/name).read_bytes() for name in files):
                raise ValueError('campaign source mismatch')
        for row in read_json(output/'input-identity.json'):
            if Path(row['original']).read_bytes() != (output/'inputs'/row['snapshot']).read_bytes():
                raise ValueError('campaign input content mismatch')
    return output


class Ledger:
    def __init__(self, output, budgets):
        self.output=Path(output);self.budgets=budgets
        if (self.output/'active-operation.json').exists():
            raise RuntimeError('unfinished operation reservation requires review')
        path=self.output/'budget.jsonl'
        self.rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    def used(self,stage):
        return sum(row['seconds'] for row in self.rows if row['stage']==stage and row['category']!='microbench')
    def call(self,stage,label,limit,operation,category='solve'):
        remaining=limit if category=='microbench' else self.budgets[stage]-self.used(stage)
        if remaining<=0:raise TimeoutError(f'{stage} budget exhausted')
        cap=min(limit,remaining)
        active=self.output/'active-operation.json'
        write_json(active,{'stage':stage,'label':label,'category':category,'reserved_seconds':cap})
        tick=time.perf_counter()
        try:return operation(cap)
        finally:
            row={'stage':stage,'label':label,'category':category,'seconds':time.perf_counter()-tick}
            with (self.output/'budget.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            self.rows.append(row);active.unlink()


def result_from_saved(saved):
    if saved['status'] in NUMERICAL:raise ArithmeticError('saved numerical failure requires review')
    return SolverResult(saved['method'],np.array(saved['parameters']),saved['status'],saved['history'],saved['costs'],saved['settings'])


class Campaign:
    def __init__(self, config, output, provenance, mode='formal', solver=minimize):
        self.config=config;self.mode=mode;self.provenance=provenance;self.solver=solver
        self.output=snapshot(output,config,provenance,mode)
        self.ledger=Ledger(self.output,{'e1':config['e1_seconds'],'e3':config['e3_seconds']})
        if mode=='formal' and not (self.output/'inherited-budget.json').exists():
            imported=[{'stage':'e1','label':entry['path'],'category':'inherited_calibration','seconds':entry['seconds']} for entry in provenance['inherited_budgets']]
            if self.ledger.rows:
                raise RuntimeError('budget ledger exists without inheritance receipt')
            with (self.output/'budget.jsonl').open('x') as stream:
                for row in imported:stream.write(json.dumps(row)+'\n')
            write_json(self.output/'inherited-budget.json',provenance['inherited_budgets'])
            self.ledger.rows.extend(imported)
        self.base=SolverConfig(**config['solver'])
        self.protocol=protocol_module()
    def solve(self,directory,label,obj,method,cfg,stage,initial=None):
        path=directory/f'{label}.json'
        if (directory/f'{label}.error.json').exists():raise ArithmeticError('saved operation exception requires review')
        if path.exists():return result_from_saved(read_json(path))
        update_status(self.output,{'state':'running','mode':self.mode,'stage':stage,'target':directory.name,'method':method,'operation':label,'solver_seconds':self.ledger.used(stage),'budget_seconds':self.ledger.budgets[stage],'metric':'both B and parameter certificate required'})
        print(f'START mode={self.mode} stage={stage} target={directory.name} operation={label} used={self.ledger.used(stage):.3f}/{self.ledger.budgets[stage]:.1f}s',flush=True)
        try:
            result=self.ledger.call(stage,f'{directory.name}/{label}',cfg.max_seconds,
                    lambda cap:self.solver(obj,method,SolverConfig(**dict(asdict(cfg),max_seconds=cap)),initial=initial))
        except (ValueError, ArithmeticError) as exc:
            write_json(directory/f'{label}.error.json',{'error':str(exc),'type':type(exc).__name__})
            raise
        if '-block' in label or label.endswith('-warmup'):
            compact=SolverResult(result.method,result.parameters,result.status,[result.history[-1]],result.costs,result.settings)
            compact.save(path)
        else:
            result.save(path)
        final=result.history[-1]
        print(f'END operation={label} status={result.status} updates={final["iteration"]} B={final["certificate_B"]:.3e} param_bound={final["parameter_error_bound"]:.3e} solve={result.costs["total_seconds"]:.4f}s',flush=True)
        if result.status in NUMERICAL:raise ArithmeticError(f'numerical failure: {label}: {result.status}')
        return result
    def calibration(self,directory,obj,stage):
        candidates=[]
        for rate in (.003,.01,.03):
            cfg=SolverConfig(**dict(asdict(self.base),learning_rate=rate))
            candidates.append((rate,self.solve(directory,f'calibration-adam-{rate:g}',obj,'adam',cfg,stage)))
        best=select_adam(candidates)
        extensions=[]
        while (extension := next_adam_rate(candidates,self.config.get('adam_extension_policy','legacy'))) is not None:
            extensions.append(extension)
            cfg=SolverConfig(**dict(asdict(self.base),learning_rate=extension))
            candidates.append((extension,self.solve(directory,f'calibration-adam-{extension:g}',obj,'adam',cfg,stage)))
        selected=select_adam(candidates)
        payload={'selected_rate':selected,'initial_best':best,'extensions':extensions,'candidate_statuses':{str(r):s.status for r,s in candidates}}
        if not (directory/'calibration.json').exists():write_json(directory/'calibration.json',payload)
        if selected is None:raise RuntimeError('all Adam candidates failed; no rate selected')
        return selected
    def timings(self,directory,obj,rate,stage):
        """Interleave methods at each warmup/measurement block; every call resets."""
        configs={m:SolverConfig(**dict(asdict(self.base),learning_rate=rate)) for m in METHODS}
        states={m:{'caps':0,'stopped':None,'blocks':[]} for m in METHODS}
        traces={};records=[]
        def one(method,label):
            state=states[method]
            result=self.solve(directory,label,obj,method,configs[method],stage)
            records.append({'method':method,'label':label,'status':result.status,'seconds':result.costs['total_seconds']})
            state['caps']=state['caps']+1 if result.status in CAPS else 0
            if state['caps']>=2:state['stopped']='two_consecutive_caps'
            return result
        for method in METHODS:
            traces[method]=one(method,f'{method}-trace')
        for method in self.protocol.rotated_method_order(1):
            if states[method]['stopped'] is None:
                warm=one(method,f'{method}-warmup')
                states[method]['block_size']=10 if warm.costs['total_seconds']<.020 else 1
        for block in range(5):
            for method in self.protocol.rotated_method_order(block):
                state=states[method]
                if state['stopped']:continue
                durations=[];successful=True
                for member in range(state['block_size']):
                    result=one(method,f'{method}-block{block}-member{member}')
                    durations.append(result.costs['total_seconds'])
                    successful=successful and result.status=='converged'
                    if state['stopped']:break
                if len(durations)==state['block_size']:
                    state['blocks'].append({'block':block,'seconds':float(np.mean(durations)),'all_converged':successful})
        summaries={}
        for method,state in states.items():
            # Failed calls remain in denominator and raw evidence, not speed rankings.
            successful=[b['seconds'] for b in state['blocks'] if b['all_converged']]
            q=np.percentile(successful,[25,75],method='linear') if successful else None
            summaries[method]={'block_size':state.get('block_size'), 'blocks':state['blocks'],'stopped':state['stopped'],
                'complete':len(state['blocks'])==5 and all(b['all_converged'] for b in state['blocks']),
                'median_seconds':float(np.median(successful)) if successful else None,'iqr_seconds':float(q[1]-q[0]) if q is not None else None,
                'calls':sum(r['method']==method for r in records),'successful_calls':sum(r['method']==method and r['status']=='converged' for r in records)}
        if not (directory/'timing.json').exists():write_json(directory/'timing.json',{'methods':summaries,'call_order':records})
        return traces,summaries
    def references(self,directory,obj,newton,stage,target,source):
        strict=SolverConfig(**dict(asdict(self.base),objective_certificate=1e-12,parameter_certificate=1e-5,max_seconds=self.config['reference_seconds']))
        refined=self.solve(directory,'reference-newton',obj,'newton',strict,stage,newton.parameters)
        final=refined.history[-1]
        if refined.status!='converged' or final['certificate_B']>1e-12 or final['parameter_error_bound']>1e-5:
            raise RuntimeError('refined Newton is not certified; evaluation unavailable')
        certified={'method':'newton_refined',**target,'parameters':refined.parameters.tolist(),
            'objective':final['objective'],'certificate_B':final['certificate_B'],'parameter_error_bound':final['parameter_error_bound'],
            'refinement_seconds':refined.costs['total_seconds'],'verified':True,'source':source,'mode':self.mode}
        if not (directory/'certified-newton.json').exists():write_json(directory/'certified-newton.json',certified)
        lbpath=directory/'reference-lbfgsb.json'
        if (directory/'reference-lbfgsb.error.json').exists():raise ArithmeticError('saved reference exception requires review')
        if not lbpath.exists():
            def operation(cap):
                tick=time.perf_counter();last={};counts={'Xv':0,'XTv':0}
                def fun(a):
                    if time.perf_counter()-tick>=cap:raise TimeoutError('reference time limit')
                    value,gradient=obj.value_gradient(a)
                    if not np.isfinite(value) or not np.isfinite(gradient).all():raise ArithmeticError('nonfinite L-BFGS-B evaluation')
                    counts['Xv']+=1;counts['XTv']+=1
                    last.update(parameters=a.tolist(),objective=value,gradient_norm=float(np.linalg.norm(gradient)))
                    return value,gradient
                try:
                    lb=scipy_minimize(fun,np.ones(obj.dimension),jac=True,method='L-BFGS-B',options={'gtol':min(np.sqrt(2*obj.regularization*1e-12),obj.regularization*1e-5)/np.sqrt(obj.dimension),'ftol':1e-15,'maxiter':20000,'maxls':60})
                    fun(lb.x)
                    last.update(success=bool(lb.success),message=str(lb.message),iterations=int(lb.nit))
                except TimeoutError as exc:last.update(success=False,message=str(exc))
                norm=last.get('gradient_norm',float('inf'))
                last.update(certificate_B=norm**2/(2*obj.regularization) if np.isfinite(norm) else None,
                    parameter_error_bound=norm/obj.regularization if np.isfinite(norm) else None,counts=counts,seconds=time.perf_counter()-tick)
                last['certified']=bool(last['certificate_B'] is not None and last['certificate_B']<=1e-12 and last['parameter_error_bound']<=1e-5)
                return last
            try:
                lb=self.ledger.call(stage,directory.name+'/reference-lbfgsb',self.config['reference_seconds'],operation,'reference')
            except (ValueError, ArithmeticError) as exc:
                write_json(directory/'reference-lbfgsb.error.json',{'error':str(exc),'type':type(exc).__name__})
                raise
            write_json(lbpath,lb)
        lb=read_json(lbpath)
        candidates=[(final['objective'],final['certificate_B'],refined.parameters,'newton_refined')]
        if lb['certified']:candidates.append((lb['objective'],lb['certificate_B'],np.array(lb['parameters']),'lbfgsb'))
        reference=min(candidates,key=lambda r:r[0])
        return refined,reference,lb
    def microbench(self,directory,obj,stage):
        path=directory/'microbench.json'
        if path.exists():return read_json(path)
        a=np.ones(obj.dimension);h=obj.hessian(a);g=obj.value_gradient(a)[1]
        operations={'gradient':lambda:obj.value_gradient(a),'hessian':lambda:obj.hessian(a),'linear_solve':lambda:linear_solve(h,-g,assume_a='pos')}
        result={}
        for name,operation in operations.items():
            self.ledger.call(stage,f'{directory.name}/microbench-{name}-warmup',60,lambda cap:operation(),'microbench') # fixed initial point
            blocks=[]
            for block in range(5):
                def measured(cap):
                    tick=time.perf_counter();count=0
                    for _ in range(10):
                        if time.perf_counter()-tick>=cap:raise TimeoutError('microbenchmark budget exhausted')
                        operation();count+=1
                    return (time.perf_counter()-tick)/count
                blocks.append(self.ledger.call(stage,f'{directory.name}/microbench-{name}-{block}',60,measured,'microbench'))
            q=np.percentile(blocks,[25,75],method='linear')
            result[name]={'block_means':blocks,'calls_per_block':10,'median_seconds':float(np.median(blocks)),'iqr_seconds':float(q[1]-q[0])}
        write_json(path,result);return result
    def target(self,seed,lam,x,rows,cache,metadata,stage,rate=None,selected_indices=None):
        n=len(x);directory=self.output/'targets'/target_id(seed,lam,n);directory.mkdir(parents=True,exist_ok=True)
        obj=PairwiseObjective(x,lam)
        if selected_indices is None:
            selected_indices=read_json(metadata).get('selected_indices')
        target={'seed':seed,'regularization':lam,'n':n}
        source={'training_cache':str(Path(cache).resolve()),'training_metadata':str(Path(metadata).resolve()),
            'run_manifest':str((self.output/'manifest.json').resolve()),'git_commit':read_json(self.output/'manifest.json')['git_commit']}
        if not (directory/'target.json').exists():write_json(directory/'target.json',{**target,'source':source,'selected_indices':selected_indices,'mode':self.mode})
        rate=rate if rate is not None else self.calibration(directory,obj,stage)
        traces,timings=self.timings(directory,obj,rate,stage)
        refined,reference,lb=self.references(directory,obj,traces['newton'],stage,target,source)
        value,bound,a,reference_method=reference
        checks=[]
        for method,result in traces.items():
            final=result.history[-1];delta=100*np.finfo(float).eps*max(1,abs(final['objective']),abs(value))
            gap=final['objective']-value
            checks.append({'method':method,'status':result.status,'iterations':final['iteration'],'objective_gap':gap,'gap_check':bool(abs(gap)<=final['certificate_B']+bound+delta),'milestones':milestones(result.history)})
        deviation=refined.parameters-1
        scale_component=float(np.mean(deviation));scale=np.full(obj.dimension,1+scale_component)
        remainder=refined.parameters-scale_component
        initial=np.ones(obj.dimension)
        diagnostic_start=time.perf_counter()
        diagnostic={'initial_spectrum':spectrum(x,initial,lam),'reference_spectrum':spectrum(x,refined.parameters,lam),
            'initial_groups':groups(x,rows,initial),'reference_groups':groups(x,rows,refined.parameters),
            'unique_queries':len({r['query_id'] for r in rows}),
            'scale_projection_squared_fraction':float(obj.dimension*scale_component**2/np.dot(deviation,deviation)) if np.dot(deviation,deviation)>0 else 0.,
            'scale_only_objective':obj.value(scale),'removed_scale_objective':obj.value(remainder),
            'interpretation':'descriptive parameter-space counterfactuals, not causal loss attribution'}
        diagnostic['seconds']=time.perf_counter()-diagnostic_start
        if not (directory/'diagnostics.json').exists():write_json(directory/'diagnostics.json',diagnostic)
        if stage=='e3':self.microbench(directory,obj,stage)
        summary={**target,'adam_rate':rate,'timing':timings,'checks':checks,'reference_method':reference_method,
            'lbfgsb_library_success':lb['success'],'lbfgsb_certified':lb['certified'],'certified_newton':str((directory/'certified-newton.json').resolve()),
            'state':'complete' if all(t['complete'] for t in timings.values()) and all(c['gap_check'] for c in checks) else 'complete_with_failures'}
        if not (directory/'summary.json').exists():write_json(directory/'summary.json',summary)
        if any(not check['gap_check'] for check in checks):
            raise ArithmeticError('objective/reference certificate consistency check failed; investigation required')
        return summary
    def run(self,stage,synthetic=None):
        if stage not in {'e1','e3'}:raise ValueError('stage must be e1 or e3')
        targets=[]
        if synthetic is not None and not isinstance(synthetic,dict):
            x,rows=synthetic
            cache=self.output/'synthetic-cache.npz';meta=self.output/'synthetic-metadata.json'
            if not cache.exists():
                with cache.open('xb') as stream:np.savez(stream,features=x)
                write_json(meta,{'synthetic':True,'triples':rows})
            targets=[(42,.01,x,rows,cache,meta,None)]
        elif synthetic is not None:
            for seed,(x,rows) in synthetic.items():
                cache=self.output/f'synthetic-cache-{seed}.npz';meta=self.output/f'synthetic-metadata-{seed}.json'
                if not cache.exists():
                    with cache.open('xb') as stream:np.savez(stream,features=x)
                    write_json(meta,{'synthetic':True,'triples':rows})
                for lam in self.provenance['frozen_lambdas']:
                    targets.append((seed,lam,x,rows,cache,meta,.01))
        elif stage=='e1':
            for seed in (42,43,44):
                entry=self.provenance['caches'][str(seed)]
                with np.load(entry['cache'],allow_pickle=False) as values:x=values['features']
                rows=read_json(entry['metadata'])['triples']
                for lam in self.provenance['frozen_grid']['lambdas']:
                    targets.append((seed,lam,x,rows,entry['cache'],entry['metadata'],self.provenance['rates'][str(lam)]))
        else:
            if not (self.output/'e1-summary.json').exists():raise RuntimeError('E3 requires completed E1 stage')
            prior=read_json(self.output/'e1-summary.json')
            if prior['stage']!='e1' or prior['mode']!='formal' or prior['state'] not in {'complete','complete_with_failures'} or len(prior['targets'])!=12:
                raise RuntimeError('E3 requires finished 12-target formal E1 stage')
            entry=self.provenance['caches']['42']
            with np.load(entry['cache'],allow_pickle=False) as values:full=values['features']
            rows=read_json(entry['metadata'])['triples']
            subsets=self.protocol.nested_subsets(list(range(len(full))))
            for n in (512,2048):
                ix=subsets[n];directory=self.output/'subsets';directory.mkdir(exist_ok=True)
                cache=directory/f'n{n}.npz';meta=directory/f'n{n}.json'
                if not cache.exists():
                    with cache.open('xb') as stream:np.savez(stream,features=full[ix])
                    write_json(meta,{'triples':[rows[i] for i in ix],'selected_indices':ix,'subset_seed':2026,'source':entry})
                targets.append((42,.001,full[ix],[rows[i] for i in ix],cache,meta,None))
        summaries=[]
        try:
            for seed,lam,x,rows,cache,meta,rate in targets:
                summaries.append(self.target(seed,lam,x,rows,cache,meta,stage,rate))
            if stage=='e3' and synthetic is None:
                # Full-size target is reused from E1; only fixed-point microbench is new.
                entry=self.provenance['caches']['42']
                with np.load(entry['cache'],allow_pickle=False) as values:full=values['features']
                directory=self.output/'targets'/target_id(42,.001,len(full))
                self.microbench(directory,PairwiseObjective(full,.001),stage)
            payload={'stage':stage,'mode':self.mode,'state':'complete' if all(s['state']=='complete' for s in summaries) else 'complete_with_failures',
                'targets':summaries,'solver_seconds':self.ledger.used(stage),'new_retrieval_evaluation':False,'old_gd_examples':self.provenance.get('old_gd_examples',[]),'microbench_seconds':sum(r['seconds'] for r in self.ledger.rows if r['stage']==stage and r['category']=='microbench'), 'diagnostic_seconds':sum(read_json(Path(s['certified_newton']).parent/'diagnostics.json')['seconds'] for s in summaries)}
            if not (self.output/f'{stage}-summary.json').exists():write_json(self.output/f'{stage}-summary.json',payload)
            export_csv(self.output,stage,payload)
            generate_figures(self.output,stage,payload)
            update_status(self.output,payload)
            print(f'CAMPAIGN STOP mode={self.mode} stage={stage} state={payload["state"]} solver_seconds={payload["solver_seconds"]:.4f} new_retrieval=False',flush=True)
            return payload
        except BaseException as exc:
            print(f'CAMPAIGN PAUSED stage={stage}: {exc}',flush=True)
            update_status(self.output,{'state':'paused','stage':stage,'mode':self.mode,'error':str(exc),'completed_targets':len(summaries),'used_seconds':self.ledger.used(stage)})
            raise


def export_csv(output,stage,payload):
    path=output/f'{stage}-timings.csv'
    if path.exists():return
    fields=['mode','seed','regularization','n','method','complete','median_seconds','iqr_seconds','calls','successful_calls','stopped']
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for target in payload['targets']:
            for method,record in target['timing'].items():
                writer.writerow({'mode':payload['mode'],**{key:target[key] for key in ('seed','regularization','n')},'method':method,**{key:record[key] for key in fields[5:]}})


def generate_figures(output,stage,payload):
    import importlib.util
    tool=Path(__file__).resolve().parents[2]/'tools/calibration_report.py'
    spec=importlib.util.spec_from_file_location('campaign_plots',tool)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    destination=output/'figures';destination.mkdir(exist_ok=True)
    for target in payload['targets']:
        directory=Path(target['certified_newton']).parent
        for axis in ('iteration','seconds'):
            path=destination/f'{directory.name}-{axis}.svg'
            if not path.exists():path.write_text(module.svg_plot(directory,axis,pattern='*-trace.json',title=f'{payload["mode"]}: {directory.name}, trace certificate (not repeated timing)'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','synthetic','run'],nargs='?',default='preflight')
    parser.add_argument('--config',type=Path,default=Path('configs/experiments.toml'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stage',choices=['e1','e3'])
    parser.add_argument('--full-grid',action='store_true',help='synthetic-only 12-target interface fixture')
    args=parser.parse_args()
    if args.full_grid and args.action!='synthetic':parser.error('--full-grid is synthetic-only')
    config=load_config(args.config)
    if args.action=='preflight':
        print(json.dumps(preflight(config),indent=2));return
    if args.output is None:parser.error('--output required for execution')
    if args.action=='run' and args.stage is None:parser.error('formal execution requires explicit --stage')
    if args.action=='synthetic':
        x=np.random.default_rng(2026).normal(scale=.1,size=(16,4))
        rows=[{'query_id':f'synthetic-{i//2}','kind':'hard' if i%2 else 'random'} for i in range(len(x))]
        config=dict(config,e1_seconds=30.,e3_seconds=30.,solver=dict(config['solver'],max_seconds=2.))
        provenance={'synthetic':True,'frozen_lambdas':[.01],'planned_targets':[{'seed':42,'regularization':.01,'n':len(x)}],'generator':'default_rng(2026), normal(scale=.1,size=(16,4))'}
        if args.full_grid:
            synthetic_data={}
            for seed in (42,43,44):
                half=np.random.default_rng(seed).normal(scale=.001,size=(8,4))
                features=np.concatenate([half,-half])
                synthetic_data[seed]=(features,rows)
            provenance['frozen_lambdas']=[.01,.001,.0001,.00001]
            provenance['planned_targets']=[{'seed':seed,'regularization':lam,'n':len(rows)} for seed in synthetic_data for lam in provenance['frozen_lambdas']]
        else:synthetic_data=(x,rows)
        campaign=Campaign(config,args.output,provenance,'synthetic')
        with (args.output/'console.log').open('a') as log,redirect_stdout(Tee(sys.stdout,log)),threadpool_limits(limits=1):campaign.run(args.stage or 'e1',synthetic_data)
    else:
        provenance=preflight(config)
        campaign=Campaign(config,args.output,provenance)
        with (args.output/'console.log').open('a') as log,redirect_stdout(Tee(sys.stdout,log)),threadpool_limits(limits=1):campaign.run(args.stage)

if __name__=='__main__':main()
