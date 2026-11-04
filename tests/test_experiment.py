import numpy as np
import pytest
from query_adapt.experiment import select_adam, milestones, spectrum, initialize
from query_adapt.solvers import SolverResult


def result(status, iterations):
    return SolverResult('adam',np.ones(2),status,[{'iteration':iterations}])


def test_candidate_selection_success_only_and_ties():
    assert select_adam([(.003,result('time_limit',2)),(.01,result('converged',4)),(.03,result('converged',4))]) == .01
    assert select_adam([(.003,result('max_iterations',2))]) is None


def test_spectrum_effective_subspace_and_milestones():
    x=np.array([[1.,0.,0.],[0.,2.,0.]])
    diagnostic=spectrum(x,np.ones(3),.01)
    assert diagnostic['rank']==diagnostic['effective_dimension']==2
    assert diagnostic['effective_threshold'] > 0
    assert diagnostic['full_condition'] > 1
    assert milestones([{'certificate_B':1.,'gradient_norm':1.,'iteration':0,'seconds':0.}])['B<=1e-09'] is None


def test_calibration_requires_freeze(tmp_path):
    from query_adapt.experiment import calibrate
    with pytest.raises(FileNotFoundError):
        calibrate({'lambdas':[.01]},tmp_path)


def test_resume_rejects_numerical_failure_and_budget(tmp_path):
    import json
    from query_adapt.experiment import calibrate
    from query_adapt.data import write_json
    (tmp_path/'diagnostics').mkdir()
    np.savez(tmp_path/'diagnostics/seed42.npz',features=np.ones((2,2)))
    write_json(tmp_path/'diagnostics/seed42.json',{'triples':[{'kind':'hard'},{'kind':'random'}]})
    write_json(tmp_path/'freeze.json',{'lambdas':[.01]})
    directory=tmp_path/'lambda-0.01';directory.mkdir()
    write_json(directory/'gd.json',{'status':'nonfinite'})
    config={'lambdas':[.01],'solver':{},'total_solver_seconds':1.0}
    with pytest.raises(ArithmeticError,match='saved numerical failure'):
        calibrate(config,tmp_path)
    (directory/'gd.json').unlink()
    (tmp_path/'budget.jsonl').write_text(json.dumps({'seconds':1.0})+'\n')
    with pytest.raises(TimeoutError,match='budget exhausted'):
        calibrate(config,tmp_path)


def test_unfinished_solve_requires_review(tmp_path):
    from query_adapt.experiment import calibrate
    from query_adapt.data import write_json
    (tmp_path/'diagnostics').mkdir()
    np.savez(tmp_path/'diagnostics/seed42.npz',features=np.ones((2,2)))
    write_json(tmp_path/'diagnostics/seed42.json',{'triples':[]})
    write_json(tmp_path/'freeze.json',{'lambdas':[.01]})
    write_json(tmp_path/'active-solve.json',{'reserved_seconds':60})
    with pytest.raises(RuntimeError,match='unfinished solve reservation'):
        calibrate({'lambdas':[.01]},tmp_path)


def test_complete_synthetic_calibration_serializes(tmp_path):
    from query_adapt.experiment import calibrate
    from query_adapt.data import write_json, read_json
    (tmp_path/'diagnostics').mkdir()
    x=np.array([[.01,0.],[0.,.01],[-.01,0.],[0.,-.01]])
    np.savez(tmp_path/'diagnostics/seed42.npz',features=x)
    write_json(tmp_path/'diagnostics/seed42.json',{'triples':[{'kind':k} for k in ['hard','hard','random','random']]})
    write_json(tmp_path/'freeze.json',{'lambdas':[.01]})
    config={'lambdas':[.01],'solver':{'objective_certificate':1e-9,'parameter_certificate':1e-3,'max_seconds':2.},'total_solver_seconds':20.,'reference_seconds':2.,'adam_rates':[.003,.01,.03]}
    calibrate(config,tmp_path)
    result=read_json(tmp_path/'summary.json')
    assert result['state']=='complete'
    assert all(isinstance(row['gap_check'],bool) for row in result['results'][0]['checks'])
    calibrate(config,tmp_path)  # Complete results resume without new solve charges.


def test_json_failure_leaves_no_partial_output(tmp_path):
    from query_adapt.data import write_json
    target=tmp_path/'invalid.json'
    with pytest.raises(TypeError):write_json(target,{'invalid':np.bool_(True)})
    assert not target.exists()


def test_adam_registered_upper03_policy_only_extends_successful_best():
    from query_adapt.experiment import next_adam_rate
    initial=[(.003,result('converged',30)),(.01,result('converged',20)),(.03,result('converged',10))]
    assert next_adam_rate(initial,'upper03')==.1
    current=initial+[(.1,result('converged',5))]
    assert next_adam_rate(current,'upper03')==.3
    assert next_adam_rate(current,'legacy') is None
    assert next_adam_rate(current+[(.3,result('converged',2))],'upper03') is None
    assert next_adam_rate(initial+[(.1,result('max_iterations',2))],'upper03') is None
    assert next_adam_rate(initial+[(.1,result('converged',10))],'upper03') is None # tie chooses .03
    assert next_adam_rate([(.003,result('converged',5)),(.01,result('converged',10)),(.03,result('converged',20))],'upper03')==.001
    assert next_adam_rate([(.003,result('max_iterations',1)),(.01,result('max_iterations',1)),(.03,result('max_iterations',1))],'upper03') is None


def test_diagnostics_reuse_has_explicit_source_and_checks_setup(tmp_path):
    from query_adapt.experiment import diagnostics
    from query_adapt.data import write_json,read_json
    source=tmp_path/'source';(source/'diagnostics').mkdir(parents=True)
    output=tmp_path/'output';output.mkdir()
    setup={'pairs':{'seed':42,'temperature':.1},'model':{'revision':'fixed'}}
    write_json(source/'manifest.json',{'setup':setup,'git_commit':'historical'})
    write_json(source/'freeze.json',{'lambdas':[.01],'seeds':[42]})
    np.savez(source/'diagnostics/seed42.npz',features=np.ones((2,2)))
    write_json(source/'diagnostics/seed42.json',{'seed':42,'pairs':setup['pairs']})
    cfg={'diagnostics_source_run':'source','lambdas':[.01],'diagnostic_seeds':[42]}
    diagnostics(cfg,tmp_path,setup,output)
    assert (output/'diagnostics/seed42.npz').read_bytes()==(source/'diagnostics/seed42.npz').read_bytes()
    assert read_json(output/'diagnostics-provenance.json')['new_spectrum_computation'] is False
    with pytest.raises(ValueError,match='setup differs'):
        diagnostics(cfg,tmp_path,dict(setup,model={'revision':'changed'}),output)


def test_calibration_inherited_budget_caps_without_double_charging(tmp_path):
    from query_adapt.experiment import calibrate
    from query_adapt.data import write_json,read_json
    root=tmp_path/'root';prior=root/'prior';prior.mkdir(parents=True)
    (prior/'budget.jsonl').write_text('{"seconds":2.0}\n')
    output=tmp_path/'output';(output/'diagnostics').mkdir(parents=True)
    write_json(output/'freeze.json',{'lambdas':[.01]})
    np.savez(output/'diagnostics/seed42.npz',features=np.ones((2,2)))
    write_json(output/'diagnostics/seed42.json',{'triples':[]})
    cfg={'lambdas':[.01],'solver':{},'inherited_budget_runs':['prior'],'total_solver_seconds':2.0}
    with pytest.raises(TimeoutError,match='budget exhausted'):
        calibrate(cfg,output,root)
    assert read_json(output/'inherited-budget.json')[0]['seconds']==2.0
    assert not (output/'budget.jsonl').exists() # Prior cost not copied into new-cost ledger.
    (prior/'budget.jsonl').write_text('{"seconds":2.1}\n')
    with pytest.raises(ValueError,match='sources differ'):
        calibrate(cfg,output,root)
