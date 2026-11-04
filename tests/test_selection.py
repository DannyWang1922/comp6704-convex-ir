import numpy as np
import pytest
from query_adapt.selection import align_queries, bootstrap_indices, paired_summary, select_systems, per_query_metrics
from query_adapt.retrieval import evaluate


def inputs(vectors):
    return {k: (['b', 'a'], v if k is None else np.tile(v, (3, 1))) for k, v in vectors.items()}


def test_alignment_validation():
    ids, a = align_queries(['b', 'a'], [[2, 1]])
    assert ids == ('a', 'b')
    np.testing.assert_array_equal(a, [[1, 2]])
    for ids, v, expected in [(['a', 'a'], [1, 2], None), (['a'], [1], ['a', 'b']), (['a'], [np.nan], None)]:
        with pytest.raises(ValueError):
            align_queries(ids, v, expected)


def test_shared_bootstrap_and_statistics():
    ix = bootstrap_indices(['a', 'b'])
    np.testing.assert_array_equal(ix, np.random.default_rng(20261101).integers(0, 2, size=(2000, 2)))
    assert not np.array_equal(ix, bootstrap_indices(['a', 'b'], 'test'))
    d = np.array([.1, -.2])
    s = paired_summary(d, ix)
    assert s['se'] == pytest.approx(d[ix].mean(axis=1).std(ddof=1))
    np.testing.assert_equal(s['ci95'], np.percentile(d[ix].mean(axis=1), [2.5, 97.5]))
    with pytest.raises(ValueError):
        bootstrap_indices(['b', 'a'])


def test_tie_preference():
    data = inputs({None: [.5, .5], 1.: [.5, .5], 2.: [.5, .5], 3.: [.5, .5], 4.: [.5, .5]})
    r = select_systems(data, bootstrap_indices(['a', 'b']))
    assert r['best'] is None and r['primary'] is None
    assert r['secondary'] == 4 and r['test_retrieval_count'] == 4


def test_primary_secondary_distinct_or_deduplicated():
    data = inputs({None: [0, 0], 1.: [.8, .8], 2.: [0, 0], 3.: [0, 0], 4.: [1., .5]})
    r = select_systems(data, bootstrap_indices(['a', 'b']))
    assert r['primary'] == 4 and r['secondary'] == 1
    assert r['test_lambdas'] == [4, 1] and r['test_retrieval_count'] == 7
    data[4.] = (['b', 'a'], np.full((3, 2), .9))
    r = select_systems(data, bootstrap_indices(['a', 'b']))
    assert r['primary'] == r['secondary'] == 4 and r['test_retrieval_count'] == 4


def test_seed_average_and_shape():
    data = inputs({None: [0, 0], 1.: [.3, .3], 2.: [.2, .2], 3.: [.1, .1], 4.: [0, 0]})
    data[1.] = (['b', 'a'], [[0, .3], [.3, .6], [.6, .9]])
    assert select_systems(data, bootstrap_indices(['a', 'b']))['means'][1.] == pytest.approx(.45)
    data[1.] = (['b', 'a'], [[0, .3]])
    with pytest.raises(ValueError):
        select_systems(data, bootstrap_indices(['a', 'b']))


def test_unrounded_metrics_match_beir_and_preserve_same_id():
    qrels = {'same': {'same': 1}, 'q2': {'relevant': 1}}
    results = {'same': {'same': 2, 'other': 1}, 'q2': {'other': 2, 'relevant': 1}}
    metrics = per_query_metrics(qrels, results)
    assert metrics['same']['nDCG@10'] == 1
    assert metrics['q2']['nDCG@10'] == pytest.approx(1 / np.log2(3))
    assert metrics['q2']['nDCG@10'] != round(metrics['q2']['nDCG@10'], 5)
    aggregate = evaluate(qrels, results)
    for metric in ('nDCG@10', 'Recall@100'):
        assert round(np.mean([x[metric] for x in metrics.values()]), 5) == aggregate[metric]
    with pytest.raises(ValueError):
        per_query_metrics(qrels, {'same': results['same']})


def test_bootstrap_query_association_is_checked():
    data = inputs({None: [.5, .5], 1.: [.5, .5], 2.: [.5, .5], 3.: [.5, .5], 4.: [.5, .5]})
    ix = bootstrap_indices(['a', 'b'])
    select_systems(data, ix, expected_query_ids=['a', 'b'])
    for wrong in (['b', 'a'], ['a', 'c']):
        with pytest.raises(ValueError, match='bootstrap query IDs'):
            select_systems(data, ix, expected_query_ids=wrong)
