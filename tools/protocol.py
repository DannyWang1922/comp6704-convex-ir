"""Planned inventory and isolated callable timing; creating plans executes nothing."""
from time import perf_counter
import numpy as np


def planned_inventory(lambdas=(1e-2, 1e-3, 1e-4, 1e-5), adam_extension_policy="upper03"):
    if len(lambdas) != 4 or len(set(lambdas)) != 4 or any(x <= 0 or not np.isfinite(x) for x in lambdas):
        raise ValueError('four distinct positive frozen lambdas required')
    if adam_extension_policy not in {"legacy", "upper03"}:
        raise ValueError("unknown Adam extension policy")
    additions=2 if adam_extension_policy=="upper03" else 1
    def config(seed, lam, n, method, **extra):
        return dict(seed=seed, regularization=lam, n=n, method=method, **extra)
    targets = [(seed, lam, 5936) for seed in (42, 43, 44) for lam in lambdas]
    targets += [(42, 1e-3, n) for n in (512, 2048)]
    conditions = [(42, lam, 5936) for lam in lambdas] + [(42, 1e-3, n) for n in (512, 2048)]
    return {
        'main': [config(*t, m) for t in targets for m in ('gd', 'newton', 'adam')],
        'old_gd': [config(42, lam, 5936, 'gd_fixed') for lam in lambdas],
        'calibration_conditions': [config(*t, 'adam') for t in conditions],
        'calibration_candidates': [config(*t, 'adam', learning_rate=lr) for t in conditions for lr in (.003, .01, .03)],
        'optional_boundary_candidates': [dict(config(*t, 'adam'), lower_learning_rate=.001, upper_learning_rate=.1, extra_upper_learning_rate=.3 if additions==2 else None, max_additions=additions) for t in conditions],
        'reference_processes': [config(*t, m) for t in targets for m in ('newton_refine', 'lbfgsb_check')],
        'unique_configuration_count': 58, 'max_unique_configuration_count': 58+6*additions,
        'ordinary_call_count': 344, 'max_ordinary_call_count': 344+6*additions, 'block_call_count':2234, 'max_block_call_count':2234+6*additions,
    }


def require_stage_authorization(action, seed=None, n=None):
    """Current authorization stops after seed-42 full-size calibration.

    Inventory entries do not authorize their execution. References are allowed
    only as part of that calibration; full-size spectra may use all three seeds.
    """
    if action == 'tools':
        return
    if action in ('spectral', 'freeze') and seed in (42, 43, 44) and n == 5936:
        return
    if action in ('calibration', 'calibration_reference') and seed == 42 and n == 5936:
        return
    raise PermissionError('current stage permits tools, full-size spectra/freeze, and seed-42 full-size calibration only')


def time_callable(solve, fresh_initial_state, *, clock=perf_counter):
    """One trace, one warmup, then 5 measurements (10 solves if warmup <20ms).

    solve(state, phase) returns a mapping with stop_reason: success,
    max_time, max_iterations, or numerical_error. Every call receives a newly
    created state, whose construction is outside timing. No I/O occurs here.
    Five block means form the median/IQR, including completed capped calls;
    incomplete blocks are retained as raw evidence but excluded from summary.
    Two consecutive capped calls stop the configuration, including trace/warmup.
    A numerical error stops immediately. Exceptions propagate for caller audit.
    """
    records, block_means = [], []
    consecutive_caps = 0
    stopped = None

    def run(phase):
        nonlocal consecutive_caps, stopped
        state = fresh_initial_state()
        start = clock()
        result = solve(state, phase)
        elapsed = clock() - start
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid clock duration')
        reason = result['stop_reason']
        if reason not in ('success', 'max_time', 'max_iterations', 'numerical_error'):
            raise ValueError('unknown stop_reason')
        records.append(dict(phase=phase, seconds=elapsed, result=result))
        consecutive_caps = consecutive_caps + 1 if reason in ('max_time', 'max_iterations') else 0
        if reason == 'numerical_error':
            stopped = 'numerical_error'
        elif consecutive_caps >= 2:
            stopped = 'two_consecutive_caps'
        return elapsed

    run('trace')
    warm = run('warmup') if stopped is None else None
    block_size = 10 if warm is not None and warm < .020 else 1
    if stopped is None:
        for block in range(5):
            elapsed = []
            for _ in range(block_size):
                elapsed.append(run('measurement'))
                if stopped is not None:
                    break
            if len(elapsed) == block_size:
                block_means.append(float(np.mean(elapsed)))
            if stopped is not None:
                break
    quartiles = np.percentile(block_means, [25, 75], method='linear') if block_means else None
    return dict(records=records, block_size=block_size, block_seconds=block_means,
                median_seconds=float(np.median(block_means)) if block_means else None,
                iqr_seconds=float(quartiles[1] - quartiles[0]) if quartiles is not None else None,
                stopped=stopped, complete=len(block_means) == 5)


def rotated_method_order(repetition, methods=('gd', 'newton', 'adam')):
    """Order for interleaving different configurations between timing blocks."""
    if not methods or repetition < 0:
        raise ValueError('nonempty methods and nonnegative repetition required')
    offset = repetition % len(methods)
    return tuple(methods[offset:]) + tuple(methods[:offset])


def nested_subsets(row_ids, sizes=(512, 2048, 5936), seed=2026):
    """Return nested row-ID lists from one seeded permutation, without data I/O."""
    ids = list(row_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('row IDs must be nonempty and unique')
    if not sizes or any(not isinstance(n, int) or n <= 0 or n > len(ids) for n in sizes) or sorted(set(sizes)) != list(sizes):
        raise ValueError('sizes must be strictly increasing positive available counts')
    order = np.random.default_rng(seed).permutation(len(ids))
    return {n: [ids[i] for i in order[:n]] for n in sizes}
