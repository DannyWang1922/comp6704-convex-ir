import pytest
from tools.protocol import planned_inventory, require_stage_authorization, time_callable, rotated_method_order


def test_inventory_and_authorization():
    p = planned_inventory()
    assert [len(p[k]) for k in ('main', 'old_gd', 'calibration_conditions', 'calibration_candidates', 'optional_boundary_candidates', 'reference_processes')] == [42, 4, 6, 18, 6, 28]
    assert p['unique_configuration_count'] == 58
    require_stage_authorization('calibration', 42, 5936)
    require_stage_authorization('spectral', 44, 5936)
    for args in [('calibration', 43, 5936), ('calibration', 42, 512), ('main', 42, 5936), ('test', 42, 5936)]:
        with pytest.raises(PermissionError):
            require_stage_authorization(*args)


def fake_run(duration, reasons=None):
    current = [0.]
    states = []
    remaining = iter(reasons) if reasons else None
    def solve(state, phase):
        assert state == {'initial': 1}
        states.append(state)
        state['initial'] = 0
        current[0] += duration
        return {'stop_reason': next(remaining) if remaining else 'success'}
    result = time_callable(solve, lambda: {'initial': 1}, clock=lambda: current[0])
    assert len({id(s) for s in states}) == len(states)
    return result


def test_regular_timing_and_short_blocks():
    r = fake_run(.05)
    assert r['complete'] and len(r['records']) == 7 and r['block_size'] == 1
    assert r['median_seconds'] == pytest.approx(.05)
    assert r['iqr_seconds'] == pytest.approx(0)
    r = fake_run(.001)
    assert r['complete'] and len(r['records']) == 52 and r['block_size'] == 10
    assert r['median_seconds'] == pytest.approx(.001)


def test_cap_and_numerical_stop_and_reset():
    r = fake_run(.05, ['max_time', 'max_iterations'])
    assert len(r['records']) == 2 and r['stopped'] == 'two_consecutive_caps'
    assert not r['complete'] and r['median_seconds'] is None
    r = fake_run(.05, ['success', 'success', 'max_time', 'success', 'max_iterations', 'max_time'])
    assert len(r['records']) == 6 and r['stopped'] == 'two_consecutive_caps'
    r = fake_run(.001, ['numerical_error'])
    assert len(r['records']) == 1 and r['stopped'] == 'numerical_error'


def test_rotation():
    assert rotated_method_order(0) == ('gd', 'newton', 'adam')
    assert rotated_method_order(1) == ('newton', 'adam', 'gd')
    assert rotated_method_order(3) == rotated_method_order(0)


def test_nested_subsets_preserve_ids_and_nesting():
    from tools.protocol import nested_subsets
    ids = ['row-' + str(i) for i in range(12)]
    result = nested_subsets(ids, (3, 8, 12))
    assert result == nested_subsets(ids, (3, 8, 12))
    assert result[3] == result[8][:3] and result[8] == result[12][:8]
    assert set(result[12]) == set(ids)
    assert len(set(result[8])) == 8
    with pytest.raises(ValueError):
        nested_subsets(['a', 'a'], (1,))
    with pytest.raises(ValueError):
        nested_subsets(ids, (8, 3))


def test_registered_extra_upper_adam_bound_and_historical_policy():
    current=planned_inventory()
    assert current['max_unique_configuration_count']==70
    assert current['max_ordinary_call_count']==356
    assert current['max_block_call_count']==2246
    assert all(c['max_additions']==2 and c['extra_upper_learning_rate']==.3 for c in current['optional_boundary_candidates'])
    legacy=planned_inventory(adam_extension_policy='legacy')
    assert legacy['max_unique_configuration_count']==64
    assert legacy['max_ordinary_call_count']==350
