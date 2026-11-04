from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pytest
from query_adapt.campaign import Campaign, Ledger, snapshot
from query_adapt.data import read_json,write_json
from query_adapt.solvers import SolverResult


def config():
    return {'e1_seconds':30.,'e3_seconds':30.,'reference_seconds':2.,'solver':{'objective_certificate':1e-9,'parameter_certificate':1e-3,'max_seconds':2.,'max_iterations':20000,'gd_step_mode':'expand'}}


def synthetic():
    x=np.random.default_rng(91).normal(scale=.1,size=(12,3))
    rows=[{'query_id':str(i//2),'kind':'hard' if i%2 else 'random'} for i in range(len(x))]
    return x,rows


def test_synthetic_full_campaign_resume_and_export(tmp_path):
    output=tmp_path/'campaign';cfg=config();p={'synthetic':True}
    c=Campaign(cfg,output,p,'synthetic')
    summary=c.run('e1',synthetic())
    assert summary['state']=='complete'
    target=summary['targets'][0];directory=Path(target['certified_newton']).parent
    certified=read_json(directory/'certified-newton.json')
    assert certified['method']=='newton_refined' and certified['verified']
    assert certified['certificate_B']<=1e-12 and certified['parameter_error_bound']<=1e-5
    timing=read_json(directory/'timing.json')
    measurements=[r['method'] for r in timing['call_order'] if '-block' in r['label']]
    size=timing['methods']['gd']['block_size']
    assert measurements[:size]==['gd']*size
    # Block 1 starts with Newton, not a second consecutive GD block.
    first_block_calls=sum(timing['methods'][m]['block_size'] for m in ('gd','newton','adam'))
    assert measurements[first_block_calls]=='newton'
    before=(output/'budget.jsonl').read_text()
    Campaign(cfg,output,p,'synthetic').run('e1',synthetic())
    assert (output/'budget.jsonl').read_text()==before
    assert (output/'e1-timings.csv').exists()
    with pytest.raises(ValueError,match='configuration'):
        Campaign(dict(cfg,e1_seconds=31),output,p,'synthetic')


def test_ledger_reservation_and_budget(tmp_path):
    write_json(tmp_path/'active-operation.json',{'label':'interrupted'})
    with pytest.raises(RuntimeError):Ledger(tmp_path,{'e1':1})
    (tmp_path/'active-operation.json').unlink()
    ledger=Ledger(tmp_path,{'e1':0})
    with pytest.raises(TimeoutError):ledger.call('e1','x',1,lambda cap:None)
    assert not (tmp_path/'active-operation.json').exists()


def test_saved_numerical_error_cannot_resume(tmp_path):
    from query_adapt.objective import PairwiseObjective
    output=tmp_path/'campaign';c=Campaign(config(),output,{'synthetic':True},'synthetic')
    directory=output/'target';directory.mkdir()
    write_json(directory/'gd-trace.json',{'status':'nonfinite'})
    with pytest.raises(ArithmeticError):c.solve(directory,'gd-trace',PairwiseObjective(np.ones((2,2)),.01),'gd',c.base,'e1')


def test_inherited_budget_import_once_and_inputs_immutable(tmp_path):
    cfg=config();source=tmp_path/'prior.jsonl';source.write_text('{"seconds":2.5}\n')
    provenance={'inherited_budgets':[{'path':str(source),'seconds':2.5}], 'caches':{}}
    output=tmp_path/'out'
    campaign=Campaign(cfg,output,provenance,'formal')
    assert campaign.ledger.used('e1')==2.5
    campaign=Campaign(cfg,output,provenance,'formal')
    assert campaign.ledger.used('e1')==2.5
    source.write_text('{"seconds":3.5}\n')
    with pytest.raises(ValueError,match='input content'):
        Campaign(cfg,output,provenance,'formal')


def test_interleaved_timing_stops_after_two_caps(tmp_path):
    from query_adapt.objective import PairwiseObjective
    calls=[]
    def capped(obj,method,cfg,initial=None):
        assert initial is None
        calls.append(method)
        return SolverResult(method,np.ones(obj.dimension),'max_iterations',
            [{'iteration':cfg.max_iterations,'objective':1.,'certificate_B':1.,'parameter_error_bound':1.,'seconds':.03,'gradient_norm':1.}],
            {'total_seconds':.03},asdict(cfg))
    campaign=Campaign(config(),tmp_path/'out',{'synthetic':True},'synthetic',solver=capped)
    directory=campaign.output/'target';directory.mkdir()
    _,timing=campaign.timings(directory,PairwiseObjective(np.ones((2,2)),.01),.01,'e1')
    assert calls==['gd','newton','adam','newton','adam','gd']
    assert all(v['calls']==2 and v['stopped']=='two_consecutive_caps' and not v['complete'] for v in timing.values())
    before=len(calls)
    campaign.timings(directory,PairwiseObjective(np.ones((2,2)),.01),.01,'e1')
    assert len(calls)==before


def test_synthetic_e3_microbench_and_diagnostics(tmp_path):
    campaign=Campaign(config(),tmp_path/'out',{'synthetic':True},'synthetic')
    result=campaign.run('e3',synthetic())
    assert result['microbench_seconds']>0
    directory=Path(result['targets'][0]['certified_newton']).parent
    benchmark=read_json(directory/'microbench.json')
    assert set(benchmark)=={'gradient','hessian','linear_solve'}
    assert all(len(b['block_means'])==5 for b in benchmark.values())
    diagnostics=read_json(directory/'diagnostics.json')
    assert diagnostics['unique_queries']==6
    assert diagnostics['seconds']>0
    assert len(list((campaign.output/'figures').glob('*.svg')))==2


def test_solver_exception_is_persisted_and_not_retried(tmp_path):
    from query_adapt.objective import PairwiseObjective
    calls=[]
    def broken(*args,**kwargs):
        calls.append(1);raise ValueError('nonfinite cached margins')
    campaign=Campaign(config(),tmp_path/'out',{'synthetic':True},'synthetic',solver=broken)
    directory=campaign.output/'target';directory.mkdir()
    obj=PairwiseObjective(np.ones((2,2)),.01)
    with pytest.raises(ValueError):campaign.solve(directory,'gd-trace',obj,'gd',campaign.base,'e1')
    with pytest.raises(ArithmeticError,match='saved operation exception'):
        campaign.solve(directory,'gd-trace',obj,'gd',campaign.base,'e1')
    assert len(calls)==1


def test_preflight_rejects_solver_drift_before_real_cache_access(tmp_path):
    from query_adapt.campaign import preflight
    run=tmp_path/'calibration';run.mkdir()
    write_json(run/'freeze.json',{'lambdas':[.01,.001,.0001,.00001],'seeds':[42,43,44]})
    write_json(run/'summary.json',{'results':[]})
    write_json(run/'manifest.json',{'config':{'solver':config()['solver']}})
    cfg=dict(config(),calibration_run=str(run),threads=1)
    cfg['solver']=dict(cfg['solver'],parameter_certificate=1.)
    with pytest.raises(ValueError,match='solver configuration'):
        preflight(cfg)
