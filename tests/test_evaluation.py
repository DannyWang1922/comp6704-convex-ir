"""Synthetic end-to-end E2 only: no SciFact paths or real metric reads."""
import json
from pathlib import Path
import numpy as np
import pytest
from query_adapt import evaluation as e


def put(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path):
    np.savez(tmp_path / 'embeddings.npz', documents=np.array([[1., 0.], [0., 1.], [-1., 0.]]),
             queries=np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
             doc_ids=np.array(['same', 'd2', 'd3']), query_ids=np.array(['train', 'same', 'v2', 't1', 't2']))
    put(tmp_path / 'splits.json', dict(train=['train'], validation=['same', 'v2'], test=['t1', 't2']))
    (tmp_path / 'train.tsv').write_text('query-id\tcorpus-id\tscore\ntrain\tsame\t1\nsame\tsame\t1\nv2\td2\t1\n')
    (tmp_path / 'test.tsv').write_text('query-id\tcorpus-id\tscore\nt1\tsame\t1\nt2\td2\t1\n')
    put(tmp_path / 'run.json', dict(mode='synthetic', git_commit='synthetic', frozen_lambdas=[.01,.001,.0001,.00001], planned_targets=[dict(seed=s, regularization=l, n=2) for s in (42,43,44) for l in (.01,.001,.0001,.00001)]))
    paths = []
    for seed in (42, 43, 44):
        np.savez(tmp_path / f'pairs-{seed}.npz', features=np.zeros((2, 2)))
        put(tmp_path / f'pairs-{seed}.json', dict(synthetic=True, triples=[dict(query_id='train')] * 2))
        for lam in (.01, .001, .0001, .00001):
            name = f'cert-{seed}-{lam}.json'
            put(tmp_path / name, dict(method='newton_refined', verified=True, seed=seed, regularization=lam,
                                      n=2, parameters=[1., 1.], certificate_B=0., parameter_error_bound=0., refinement_seconds=.1,
                                      source=dict(training_cache=str(tmp_path / f'pairs-{seed}.npz'), training_metadata=str(tmp_path / f'pairs-{seed}.json'), run_manifest=str(tmp_path / 'run.json'), git_commit='synthetic')))
            paths.append(name)
    manifest = tmp_path / 'manifest.json'
    put(manifest, dict(synthetic=True, embeddings='embeddings.npz', splits='splits.json', train_qrels='train.tsv', test_qrels='test.tsv', certificates=paths))
    return manifest, tmp_path / 'evaluation'


def test_complete_validation_freeze_test_summary(fixture, monkeypatch):
    manifest, output = fixture
    original = e.read_qrels
    reads = []
    def watched(path):
        reads.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(e, 'read_qrels', watched)
    validation = e.validate(manifest, output)
    assert 'test.tsv' not in reads
    assert len(validation['systems']) == 13
    assert validation['systems'][0]['metrics']['nDCG@10'] == [1., 1.]
    frozen = e.freeze(manifest, output)
    assert 'test.tsv' not in reads
    assert frozen['primary'] is None and frozen['secondary'] == .01
    result = e.test(manifest, output)
    assert len(result['systems']) == 4 and 'test.tsv' in reads
    summary = e.summary(manifest, output)
    assert summary['retrieval_count'] == 4 and summary['query_count'] == 2
    with np.load(output / 'test-bootstrap.npz') as c:
        np.testing.assert_array_equal(c['indices'], e.bootstrap_indices(['t1', 't2'], 'test'))
        assert c['seed'] == 20261102
    assert (output / 'summary.json').exists()
    with pytest.raises(FileExistsError):
        e.test(manifest, output)
    with pytest.raises(FileExistsError):
        e.validate(manifest, output)


def test_no_freeze_rejects_before_test_read(fixture, monkeypatch):
    manifest, output = fixture
    e.validate(manifest, output)
    def forbidden(*args):
        raise AssertionError('test qrels read before freeze')
    monkeypatch.setattr(e, 'read_qrels', forbidden)
    with pytest.raises(FileNotFoundError):
        e.test(manifest, output)


@pytest.mark.parametrize('change', ['parameter', 'method', 'missing', 'duplicate', 'heldout', 'qrels', 'ids'])
def test_bad_inputs_rejected(fixture, change):
    manifest, output = fixture
    m = json.loads(manifest.read_text())
    cp = manifest.parent / m['certificates'][0]
    c = json.loads(cp.read_text())
    if change == 'parameter':
        c['parameters'] = [2., 1.]
        put(cp, c)
    elif change == 'method':
        c['method'] = 'lbfgsb'
        put(cp, c)
    elif change == 'missing':
        m['certificates'].pop()
        put(manifest, m)
    elif change == 'duplicate':
        m['certificates'][-1] = m['certificates'][0]
        put(manifest, m)
    elif change == 'heldout':
        p = manifest.parent / 'pairs-42.json'
        meta = json.loads(p.read_text())
        meta['triples'][0]['query_id'] = 'same'
        put(p, meta)
    elif change == 'qrels':
        with (manifest.parent / 'train.tsv').open('a') as s:
            s.write('same\tsame\t1\n')
    else:
        p = manifest.parent / 'splits.json'
        splits = json.loads(p.read_text())
        splits['validation'] = ['same', 'same']
        put(p, splits)
    with pytest.raises(ValueError):
        e.validate(manifest, output)


def test_modified_input_or_freeze_rejected(fixture):
    manifest, output = fixture
    e.validate(manifest, output)
    e.freeze(manifest, output)
    m = json.loads(manifest.read_text())
    p = manifest.parent / m['certificates'][0]
    c = json.loads(p.read_text())
    c['refinement_seconds'] = 2.
    put(p, c)
    with pytest.raises(ValueError, match='inputs or protocol changed'):
        e.test(manifest, output)


def test_frozen_identity_cannot_change(fixture):
    manifest, output = fixture
    e.validate(manifest, output)
    e.freeze(manifest, output)
    p = output / 'freeze.json'
    f = json.loads(p.read_text())
    f['primary'] = .001
    put(p, f)
    with pytest.raises(ValueError, match='identity'):
        e.test(manifest, output)


def test_real_synthetic_rankings_keep_distinct_primary_secondary(fixture):
    manifest, output = fixture
    root = manifest.parent
    m = json.loads(manifest.read_text())
    # Synthetic logistic objectives have a unique scalar stationary point.
    roots = {}
    for lam in (.01, .001, .0001, .00001):
        lo, hi = 1., 30.
        for _ in range(100):
            a = (lo + hi) / 2
            if lam * (a - 1) - 1 / (1 + np.exp(a)) > 0:
                hi = a
            else:
                lo = a
        roots[lam] = (lo + hi) / 2
    low = (1 + roots[.01]) / 2
    middle = (roots[.01] + roots[.001]) / 2
    high = (roots[.001] + roots[.0001]) / 2
    ratios = [low] * 3 + [middle] * 3 + [high] * 2
    relevant = ['same'] * 3 + ['same', 'same', 'd2'] + ['d2'] * 2
    vids = [f'v{i}' for i in range(8)]
    vectors = np.array([[1., r] for r in ratios])
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    np.savez(root / 'embeddings.npz', documents=np.array([[1., 0.], [0., 1.], [-1., 0.]]),
             queries=np.vstack(([1., 0.], vectors, [1., 0.], [0., 1.])),
             doc_ids=np.array(['same', 'd2', 'd3']), query_ids=np.array(['train'] + vids + ['t1', 't2']))
    put(root / 'splits.json', dict(train=['train'], validation=vids, test=['t1', 't2']))
    (root / 'train.tsv').write_text('query-id\tcorpus-id\tscore\ntrain\tsame\t1\n' + ''.join(f'{q}\t{d}\t1\n' for q, d in zip(vids, relevant)))
    for seed in (42, 43, 44):
        np.savez(root / f'pairs-{seed}.npz', features=np.array([[1., 0.], [1., 0.]]))
    for name in m['certificates']:
        p = root / name
        c = json.loads(p.read_text())
        lam = c['regularization']
        c['parameters'] = [roots[lam], 1.]
        residual = abs(lam * (roots[lam] - 1) - 1 / (1 + np.exp(roots[lam])))
        c['certificate_B'] = residual**2 / (2 * lam)
        c['parameter_error_bound'] = residual / lam
        put(p, c)
    e.validate(manifest, output)
    frozen = e.freeze(manifest, output)
    assert frozen['primary'] == .01 and frozen['secondary'] == .001
    assert frozen['test_retrieval_count'] == 7
    result = e.test(manifest, output)
    assert len(result['systems']) == 7
    summary = e.summary(manifest, output)
    assert not summary['merged'] and summary['retrieval_count'] == 7


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_resume_reuses_completed_systems_without_overwrite(fixture, monkeypatch, split):
    manifest, output = fixture
    original = e.retrieve
    if split == 'test':
        e.validate(manifest, output)
        e.freeze(manifest, output)
    calls = [0]
    def interrupted(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError('synthetic interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(e, 'retrieve', interrupted)
    entry = e.validate if split == 'validation' else e.test
    with pytest.raises(RuntimeError, match='interruption'):
        entry(manifest, output)
    paths = sorted(output.glob(f'{split}-system-*.json'))
    original_bytes = {p: p.read_bytes() for p in paths}
    monkeypatch.setattr(e, 'retrieve', original)
    result = entry(manifest, output)
    assert len(result['systems']) == (13 if split == 'validation' else 4)
    assert all(p.read_bytes() == content for p, content in original_bytes.items())


@pytest.mark.parametrize('change', ['grid', 'target', 'revision'])
def test_campaign_manifest_semantics(fixture, change):
    manifest, output = fixture
    p = manifest.parent / 'run.json'
    run = json.loads(p.read_text())
    if change == 'grid':
        run['frozen_lambdas'] = [.1, .001, .0001, .00001]
    elif change == 'target':
        run['planned_targets'].pop(0)
    else:
        run['git_commit'] = 'different'
    put(p, run)
    with pytest.raises(ValueError):
        e.validate(manifest, output)


def test_manifest_builder_uses_completed_campaign_without_test_read(fixture):
    manifest, output = fixture
    root = manifest.parent
    original = json.loads(manifest.read_text())
    # Campaign manifest and E2 input manifest are distinct files.
    (root / 'manifest.json').rename(root / 'original-e2.json')
    (root / 'run.json').rename(root / 'manifest.json')
    for name in original['certificates']:
        p = root / name
        cert = json.loads(p.read_text())
        cert['source']['run_manifest'] = str(root / 'manifest.json')
        put(p, cert)
    put(root / 'e1-summary.json', dict(stage='e1', mode='synthetic', state='complete', targets=[dict(certified_newton=name) for name in original['certificates']]))
    (root / 'test.tsv').unlink()  # builder cannot depend on any held-out contents
    record = e.build_manifest(root, root / 'built.json', root / 'embeddings.npz', root / 'splits.json', root / 'train.tsv', root / 'test.tsv', synthetic=True)
    assert len(record['certificates']) == 12
    assert all(Path(p).is_absolute() for p in record['certificates'])


def test_formal_requires_completed_e1_even_with_valid_certificate(fixture):
    manifest, output = fixture
    root = manifest.parent
    m = json.loads(manifest.read_text())
    m['synthetic'] = False
    put(manifest, m)
    run_path = root / 'run.json'
    run = json.loads(run_path.read_text())
    run['mode'] = 'formal'
    run['provenance'] = dict(splits=str(root / 'splits.json'), caches={str(seed): dict(cache=str(root / f'pairs-{seed}.npz'), metadata=str(root / f'pairs-{seed}.json')) for seed in (42,43,44)})
    put(run_path, run)
    with pytest.raises(ValueError, match='completed full E1'):
        e.validate(manifest, output)
