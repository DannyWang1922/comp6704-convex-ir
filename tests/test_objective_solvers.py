import numpy as np
import pytest
import torch
from scipy.optimize import minimize as scipy_minimize
from query_adapt.objective import PairwiseObjective
from query_adapt.solvers import SolverConfig, minimize


@pytest.fixture
def objective():
    return PairwiseObjective(np.random.default_rng(7).normal(size=(40, 6)), 0.2)


def test_derivatives_against_autograd_and_finite_difference(objective):
    a = np.random.default_rng(3).normal(size=6)
    x = torch.tensor(objective.features, dtype=torch.float64)
    at = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    def function(t):
        return torch.nn.functional.softplus(-(x @ t)).mean() + 0.1 * ((t - 1)**2).sum()
    loss = function(at)
    gradient = torch.autograd.grad(loss, at)[0].detach().numpy()
    hessian = torch.autograd.functional.hessian(function, at).numpy()
    value, analytic = objective.value_gradient(a)
    assert value == pytest.approx(loss.item(), abs=1e-12)
    np.testing.assert_allclose(analytic, gradient, atol=1e-12)
    np.testing.assert_allclose(objective.hessian(a), hessian, atol=1e-12)
    eps = 1e-5
    numeric = [(objective.value(a + eps * e) - objective.value(a - eps * e)) / (2 * eps)
               for e in np.eye(6)]
    np.testing.assert_allclose(analytic, numeric, atol=1e-9)
    assert np.linalg.eigvalsh(hessian).min() >= objective.regularization - 1e-12


@pytest.mark.parametrize("method", ["gd", "newton", "adam"])
def test_solvers_against_independent_reference(objective, method):
    # Independent expression and finite-difference Jacobian avoid reusing the
    # implementation's derivatives in the reference solve.
    def reference_loss(a):
        margins = objective.features @ a
        return (np.mean(np.logaddexp(0, -margins))
                + objective.regularization * np.sum(np.square(a - np.ones(6))) / 2)
    reference = scipy_minimize(reference_loss, np.ones(6), jac="3-point",
                               method="BFGS", options={"gtol": 1e-9})
    assert reference.success, reference.message
    result = minimize(objective, method, SolverConfig(gradient_tolerance=1e-7,
                      max_seconds=20, max_iterations=20000, learning_rate=0.03))
    assert result.status == "converged", result.summary()
    assert result.history[-1]["gradient_norm"] <= 1e-7
    np.testing.assert_allclose(result.parameters, reference.x, atol=2e-6)
    assert result.history[-1]["objective"] == pytest.approx(reference.fun, abs=1e-11)
    assert result.costs["total_seconds"] >= result.history[-1]["seconds"]
    if method != "adam":
        assert np.all(np.diff([row["objective"] for row in result.history]) <= 1e-12)
        assert all(row["objective"] <= row["armijo_bound"] + 1e-14
                   for row in result.history[1:])
    else:
        assert all(row["armijo_bound"] is None for row in result.history)


def test_stops_are_explicit(objective, tmp_path):
    result = minimize(objective, "gd", SolverConfig(max_iterations=0))
    assert result.status == "max_iterations"
    path = tmp_path / "synthetic-run.json"
    result.save(path)
    import json
    saved = json.loads(path.read_text())
    assert saved["status"] == "max_iterations"
    assert saved["history"] == result.history
    assert saved["settings"]["max_iterations"] == 0
    with pytest.raises(FileExistsError):
        result.save(path)
    assert minimize(objective, "gd", SolverConfig(max_seconds=1e-12)).status == "time_limit"
    with pytest.raises(ValueError):
        minimize(objective, "unknown")
    with pytest.raises(ValueError):
        SolverConfig(learning_rate=-1)


def test_extreme_logits_remain_finite():
    obj = PairwiseObjective(np.array([[1000., -1000.], [-1000., 1000.]]), 0.1)
    a = np.array([1000., -1000.])
    value, gradient = obj.value_gradient(a)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert np.isfinite(obj.hessian(a)).all()


def test_line_search_failure_is_reported():
    # A high-curvature objective rejects the first unit GD step.
    obj = PairwiseObjective(np.eye(2) * 100, 100)
    result = minimize(obj, "gd", SolverConfig(max_backtracks=1), initial=np.zeros(2))
    assert result.status == "line_search_failed"
    assert result.costs["line_search_evaluations"] == 1
    np.testing.assert_array_equal(result.parameters, np.zeros(2))


def test_live_callback_does_not_change_result(objective):
    observed = []
    config = SolverConfig(max_iterations=3)
    expected = minimize(objective, 'gd', config)
    actual = minimize(objective, 'gd', config, callback=observed.append)
    np.testing.assert_array_equal(actual.parameters, expected.parameters)
    assert len(observed) == len(actual.history) == 4
    assert observed[-1]['objective'] == actual.history[-1]['objective']
    observed[0]['objective'] = -999
    assert actual.history[0]['objective'] != -999


def test_expansion_cache_counts_and_certificates(objective):
    result = minimize(objective, 'gd', SolverConfig(max_iterations=3, gradient_tolerance=1e-30))
    assert result.history[1]['trial_step_size'] == 1
    for previous, row in zip(result.history[1:], result.history[2:]):
        assert row['trial_step_size'] == 2*previous['step_size']
    assert result.costs['Xv'] == 1 + result.costs['line_search_evaluations']
    assert result.costs['XTv'] == len(result.history)
    for row in result.history:
        assert row['certificate_B'] == pytest.approx(row['gradient_norm']**2/(2*objective.regularization))
    cfg = SolverConfig(objective_certificate=1e-9, parameter_certificate=1e-3)
    result = minimize(objective, 'newton', cfg)
    assert result.status == 'converged'
    assert result.history[-1]['certificate_B'] <= 1e-9
    assert result.history[-1]['parameter_error_bound'] <= 1e-3
    assert result.costs['hessian_builds'] == result.costs['linear_solves'] == result.history[-1]['iteration']


def test_unit_mode_preserves_original_updates(objective):
    a = np.ones(objective.dimension)
    for _ in range(4):
        value, gradient = objective.value_gradient(a)
        step = 1.0
        while objective.value(a-step*gradient) > value-1e-4*step*np.dot(gradient,gradient):
            step *= .5
        a -= step*gradient
    result = minimize(objective,'gd',SolverConfig(max_iterations=4,gd_step_mode='unit',gradient_tolerance=1e-30))
    np.testing.assert_array_equal(result.parameters,a)


def test_cached_margins_validate_shape_and_finite(objective):
    with pytest.raises(ValueError):
        objective.gradient_from_margins(np.ones(6),np.ones(2))
    with pytest.raises(ValueError):
        objective.hessian_from_margins(np.full(40,np.nan))


class DominantQuadratic:
    """Exact SPD quadratic exposing the near-2/L sign-changing GD mode."""
    dimension = 2
    regularization = .01
    curvature = np.array([.01, 1.99])
    def __init__(self):
        self.accepted=[]
    def _parameter(self,a):
        return np.asarray(a,dtype=float)
    def margins(self,a):
        return np.asarray(a,dtype=float).copy()
    def value_from_margins(self,a,z):
        return float(.5*np.dot(a,self.curvature*a))
    def gradient_from_margins(self,a,z):
        self.accepted.append(a.copy())
        return self.curvature*a
    def hessian_from_margins(self,z):
        return np.diag(self.curvature)


def test_dominant_quadratic_main_gd_avoids_near_boundary_mode():
    old,new=DominantQuadratic(),DominantQuadratic()
    base=dict(max_iterations=3000,gradient_tolerance=1e-10,max_seconds=10.)
    previous=minimize(old,'gd',SolverConfig(**base,gd_armijo=1e-4),initial=np.array([0.,1.]))
    current=minimize(new,'gd',SolverConfig(**base,gd_armijo=.5),initial=np.array([0.,1.]))
    assert previous.status==current.status=='converged'
    old_values=np.array(old.accepted)[:,1];new_values=np.array(new.accepted)[:,1]
    assert np.sum(old_values[1:]*old_values[:-1]<0)>100
    assert np.all(new_values>=0)
    assert current.history[-1]['iteration'] < previous.history[-1]['iteration']/100
    # Doubling the accepted .5 step proposes 1 again, then rejects it once.
    assert all(row['backtracks']==1 for row in current.history[1:])
    assert all(row['trial_step_size']==1 for row in current.history[1:])
    assert all(row['step_size']==.5 for row in current.history[1:])
    for before,after in zip(current.history,current.history[1:]):
        expected=before['objective']-.5*after['step_size']*before['gradient_norm']**2
        assert after['armijo_bound']==pytest.approx(expected)
        assert after['objective']<=after['armijo_bound']
    assert current.settings['effective_armijo']==.5


@pytest.mark.parametrize('method,mode,expected',[('gd','expand',.5),('gd','unit',1e-4),('newton','expand',1e-4),('adam','expand',None)])
def test_method_specific_armijo_isolation(objective,method,mode,expected):
    cfg=SolverConfig(gd_step_mode=mode,max_iterations=3)
    result=minimize(objective,method,cfg)
    assert result.settings['effective_armijo']==expected
    assert all(row['effective_armijo']==expected for row in result.history)
    if method in {'gd','newton'}:
        assert all(row['objective']<=row['armijo_bound']+1e-14 for row in result.history[1:])
    if method!='gd' or mode=='unit':
        original=minimize(objective,method,SolverConfig(gd_step_mode=mode,max_iterations=3,gd_armijo=1e-4))
        np.testing.assert_array_equal(original.parameters,result.parameters)
