"""Three full-batch solvers; no dataset loading or automatic experiment runs."""
from dataclasses import asdict, dataclass, field
from time import perf_counter
import numpy as np
from scipy.linalg import solve as linear_solve
from .data import write_json
from .objective import PairwiseObjective


@dataclass(frozen=True)
class SolverConfig:
    # Numerical defaults for API use/tests, not selected experiment settings.
    gradient_tolerance: float = 1e-6
    objective_certificate: float | None = None
    parameter_certificate: float | None = None
    gd_step_mode: str = "expand"
    max_iterations: int = 10000
    max_seconds: float = 60.0
    armijo: float = 1e-4
    gd_armijo: float = 0.5
    shrink: float = 0.5
    max_backtracks: int = 60
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.gd_step_mode not in {"expand", "unit"}:
            raise ValueError("invalid GD step mode")
        for bound in (self.objective_certificate, self.parameter_certificate):
            if bound is not None and (not np.isfinite(bound) or bound <= 0):
                raise ValueError("certificate bounds must be positive")
        if (self.objective_certificate is None) != (self.parameter_certificate is None):
            raise ValueError("both certificate bounds are required")
        values = [self.gradient_tolerance, self.max_seconds, self.learning_rate, self.epsilon]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("tolerance, time, learning rate and epsilon must be positive")
        if self.max_iterations < 0 or self.max_backtracks < 1:
            raise ValueError("invalid iteration limits")
        if not (0 < self.armijo < 1 and 0 < self.gd_armijo < 1 and 0 < self.shrink < 1):
            raise ValueError("invalid line-search constants")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("invalid Adam moments")


@dataclass
class SolverResult:
    method: str
    parameters: np.ndarray
    status: str
    history: list[dict] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def summary(self):
        return {"method": self.method, "status": self.status,
                "final": self.history[-1], "costs": self.costs}

    def save(self, path):
        """Save parameters, settings and every iterate without overwriting a file.

        Nonfinite diagnostic values are stored as null; status retains the failure.
        """
        history = [{key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                    for key, value in row.items()} for row in self.history]
        write_json(path, {"method": self.method, "status": self.status,
                          "parameters": self.parameters.tolist(), "settings": self.settings,
                          "history": history, "costs": self.costs})


def minimize(objective: PairwiseObjective, method: str, config=None, initial=None, callback=None):
    """Run one explicitly requested solve, returning every accepted iterate.

    Time limits are checked between operations and during backtracking. A single
    BLAS operation cannot be interrupted, so the limit may be exceeded by one call.
    """
    if method not in {"gd", "newton", "adam"}:
        raise ValueError("method must be gd, newton or adam")
    cfg = config or SolverConfig()
    effective_armijo = (cfg.gd_armijo if method == "gd" and cfg.gd_step_mode == "expand" else cfg.armijo) if method != "adam" else None
    a = np.ones(objective.dimension) if initial is None else np.array(initial, dtype=float, copy=True)
    objective._parameter(a)
    moment = np.zeros_like(a)
    variance = np.zeros_like(a)
    costs = {"value_gradient_seconds": 0.0, "hessian_seconds": 0.0,
             "linear_solve_seconds": 0.0, "line_search_seconds": 0.0,
             "line_search_evaluations": 0, "Xv": 0, "XTv": 0,
             "hessian_builds": 0, "linear_solves": 0}
    history = []
    start = perf_counter()
    step_size = 0.0
    backtracks = 0
    armijo_bound = None
    status = "max_iterations"
    cached_margins = None
    trial_step_size = 0.0
    for iteration in range(cfg.max_iterations + 1):
        tick = perf_counter()
        if cached_margins is None:
            cached_margins = objective.margins(a)
            costs["Xv"] += 1
        value = objective.value_from_margins(a, cached_margins)
        gradient = objective.gradient_from_margins(a, cached_margins)
        costs["XTv"] += 1
        costs["value_gradient_seconds"] += perf_counter() - tick
        norm = float(np.linalg.norm(gradient))
        elapsed = perf_counter() - start
        certificate = float(np.square(np.float64(norm)) / (2 * objective.regularization))
        distance = norm / objective.regularization
        regularizer = float(0.5 * objective.regularization * np.sum((a - 1)**2))
        history.append({"iteration": iteration, "seconds": elapsed, "objective": value,
                        "gradient_norm": norm, "step_size": step_size, "backtracks": backtracks,
                        "armijo_bound": armijo_bound, "effective_armijo": effective_armijo, "trial_step_size": trial_step_size,
                        "certificate_B": certificate, "parameter_error_bound": distance,
                        "ranking_loss": value - regularizer, "regularizer": regularizer,
                        "parameter_distance": float(np.linalg.norm(a - 1))})
        if callback is not None:
            callback(dict(history[-1]))
        if not (np.isfinite(value) and np.isfinite(norm)):
            status = "nonfinite"
            break
        passed = (norm <= cfg.gradient_tolerance if cfg.objective_certificate is None else
                  certificate <= cfg.objective_certificate and distance <= cfg.parameter_certificate)
        if passed:
            status = "converged"
            break
        if elapsed >= cfg.max_seconds:
            status = "time_limit"
            break
        if iteration == cfg.max_iterations:
            break
        if method == "adam":
            armijo_bound = None
            t = iteration + 1
            moment = cfg.beta1 * moment + (1 - cfg.beta1) * gradient
            variance = cfg.beta2 * variance + (1 - cfg.beta2) * gradient**2
            direction = -(moment / (1 - cfg.beta1**t)) / (np.sqrt(variance / (1 - cfg.beta2**t)) + cfg.epsilon)
            step_size, backtracks = cfg.learning_rate, 0
        else:
            direction = -gradient
            if method == "newton":
                tick = perf_counter()
                hessian = objective.hessian_from_margins(cached_margins)
                costs["hessian_builds"] += 1
                costs["hessian_seconds"] += perf_counter() - tick
                if not np.isfinite(hessian).all():
                    status = "nonfinite_hessian"
                    break
                if perf_counter() - start >= cfg.max_seconds:
                    status = "time_limit"
                    break
                tick = perf_counter()
                try:
                    costs["linear_solves"] += 1
                    direction = linear_solve(hessian, -gradient, assume_a="pos")
                except np.linalg.LinAlgError:
                    status = "linear_solve_failed"
                    break
                costs["linear_solve_seconds"] += perf_counter() - tick
            slope = float(gradient @ direction)
            if not np.isfinite(slope) or slope >= 0:
                status = "invalid_direction"
                break
            tick = perf_counter()
            step_size = (2 * step_size if method == "gd" and cfg.gd_step_mode == "expand"
                         and iteration > 0 else 1.0)
            trial_step_size = step_size
            accepted = False
            for backtracks in range(cfg.max_backtracks):
                if perf_counter() - start >= cfg.max_seconds:
                    status = "time_limit"
                    break
                candidate = a + step_size * direction
                if np.isfinite(candidate).all():
                    candidate_margins = objective.margins(candidate)
                    costs["Xv"] += 1
                    trial = objective.value_from_margins(candidate, candidate_margins)
                    costs["line_search_evaluations"] += 1
                    if np.isfinite(trial) and trial <= value + effective_armijo * step_size * slope:
                        armijo_bound = float(value + effective_armijo * step_size * slope)
                        cached_margins = candidate_margins
                        accepted = True
                        break
                step_size *= cfg.shrink
            costs["line_search_seconds"] += perf_counter() - tick
            if not accepted:
                if status != "time_limit":
                    status = "line_search_failed"
                break
        candidate = a + step_size * direction
        if not np.isfinite(candidate).all():
            status = "nonfinite_update"
            break
        a = candidate
        if method == "adam":
            cached_margins = None
    costs["total_seconds"] = perf_counter() - start
    settings = dict(asdict(cfg), effective_armijo=effective_armijo)
    return SolverResult(method, a, status, history, costs, settings)
