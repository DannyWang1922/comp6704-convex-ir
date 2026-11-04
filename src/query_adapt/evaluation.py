"""Explicit E2 validation, freeze and test entry points. Never runs on import.

CLI: python -m query_adapt.evaluation {validate,freeze,test,summary}
     --manifest manifest.json --output new-e2-directory
Manifest paths are relative to the manifest: embeddings, splits, train_qrels,
 test_qrels, certificates (12 JSON paths), optional synthetic (default false).
Certificates follow campaign's certified-newton.json schema.
"""
import argparse
import hashlib
import json
from time import perf_counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .data import read_json, read_qrels, write_json, validate_embeddings
from .objective import PairwiseObjective
from .retrieval import retrieve, evaluate
from .selection import bootstrap_indices, paired_summary, per_query_metrics, select_systems

METRICS = ('nDCG@10', 'Recall@100')
RULE = dict(version=1, metric='nDCG@10', seeds=[42, 43, 44], bootstrap=2000,
            validation_seed=20261101, test_seed=20261102, rng='PCG64', ddof=1,
            percentile_method='linear', preference='baseline_then_largest_lambda',
            certificate_B=1e-12, parameter_error_bound=1e-5)


def _digest(path):
    # Freeze compatibility requires exact content identity, not mtimes alone.
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _path(base, value):
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def _load(manifest_path):
    path = Path(manifest_path).resolve()
    m = read_json(path)
    base = path.parent
    paths = {key: _path(base, m[key]) for key in ('embeddings', 'splits', 'train_qrels', 'test_qrels')}
    sources = {str(path): _digest(path)}
    def remember(p):
        if str(p) not in sources:
            sources[str(p)] = _digest(p)
    for key in ('embeddings', 'splits', 'train_qrels'):
        remember(paths[key])
    splits = read_json(paths['splits'])
    for split in ('train', 'validation', 'test'):
        ids = splits[split]
        if not ids or any(not isinstance(q, str) for q in ids) or len(set(ids)) != len(ids):
            raise ValueError('split IDs must be nonempty unique strings')
    if any(set(splits[a]) & set(splits[b]) for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('query partitions overlap')
    with np.load(paths['embeddings'], allow_pickle=False) as c:
        docs, queries = c['documents'], c['queries']
        dids, qids = c['doc_ids'].tolist(), c['query_ids'].tolist()
    validate_embeddings(docs, queries)
    if len(dids) != len(docs) or len(qids) != len(queries) or len(set(dids)) != len(dids) or len(set(qids)) != len(qids):
        raise ValueError('duplicate or misaligned embedding IDs')
    if any(not isinstance(x, str) for x in dids + qids) or not set(sum([splits[k] for k in ('train', 'validation', 'test')], [])) <= set(qids):
        raise ValueError('split queries absent from cache')
    train = read_qrels(paths['train_qrels'])
    if set(train) != set(splits['train']) | set(splits['validation']):
        raise ValueError('training qrels do not match original train/validation split')
    _check_qrels(train, dids)
    certs = {}
    for value in m['certificates']:
        cp = _path(base, value)
        cert = read_json(cp)
        source = cert['source']
        extra_paths = [_path(cp.parent, source[k]) for k in ('training_cache', 'training_metadata', 'run_manifest')]
        for p in [cp] + extra_paths:
            remember(p)
        if cert['method'] != 'newton_refined' or cert['verified'] is not True:
            raise ValueError('only verified refined Newton parameters may be evaluated')
        run = read_json(extra_paths[2])
        expected_mode = 'synthetic' if m.get('synthetic', False) else 'formal'
        target = dict(seed=cert['seed'], regularization=cert['regularization'], n=cert['n'])
        if run.get('mode') != expected_mode or run.get('git_commit') != source.get('git_commit') or not any(all(t.get(k) == v for k, v in target.items()) for t in run.get('planned_targets', [])):
            raise ValueError('certificate is not a declared campaign target/revision')
        if expected_mode == 'formal':
            provenance = run['provenance']
            entry = provenance['caches'][str(cert['seed'])]
            if [_path(extra_paths[2].parent, entry[k]) for k in ('cache', 'metadata')] != extra_paths[:2] or _path(extra_paths[2].parent, provenance['splits']) != paths['splits']:
                raise ValueError('certificate cache or split differs from campaign provenance')
        if expected_mode == 'formal':
            summary_path = extra_paths[2].parent / 'e1-summary.json'
            if not summary_path.is_file():
                raise ValueError('formal E2 requires completed full E1 summary')
            remember(summary_path)
            completed = read_json(summary_path)
            completed_targets = completed.get('targets', [])
            expected_targets = {(s, l, 5936) for s in RULE['seeds'] for l in run['frozen_lambdas']}
            actual_targets = {(t['seed'], t['regularization'], t['n']) for t in completed_targets}
            if completed.get('stage') != 'e1' or completed.get('mode') != 'formal' or completed.get('state') not in ('complete', 'complete_with_failures') or len(completed_targets) != 12 or actual_targets != expected_targets:
                raise ValueError('formal E2 requires completed full E1 summary')
            matching = [t for t in completed_targets if all(t.get(k) == v for k, v in target.items())]
            if len(matching) != 1 or _path(summary_path.parent, matching[0]['certified_newton']) != cp:
                raise ValueError('certificate differs from completed campaign result')
        if cert['regularization'] not in run.get('frozen_lambdas', []):
            raise ValueError('certificate lambda differs from frozen campaign grid')
        if not source.get('git_commit'):
            raise ValueError('certificate needs source revision')
        seed, lam, n = cert['seed'], cert['regularization'], cert['n']
        if seed not in RULE['seeds'] or not np.isfinite(lam) or lam <= 0 or (seed, lam) in certs:
            raise ValueError('invalid or duplicate certificate target')
        if not m.get('synthetic', False) and n != 5936:
            raise ValueError('formal E2 requires complete 5936-row training objectives')
        meta = read_json(extra_paths[1])
        rows = meta['triples']
        if m.get('synthetic', False) and meta.get('synthetic') is not True:
            raise ValueError('synthetic manifest requires explicitly synthetic cache metadata')
        if len(rows) != n or any(r['query_id'] not in set(splits['train']) for r in rows):
            raise ValueError('certificate training rows contaminate held-out queries')
        with np.load(extra_paths[0], allow_pickle=False) as c:
            features = c['features']
        if features.shape != (n, docs.shape[1]):
            raise ValueError('certificate objective/cache dimensions differ')
        a = np.asarray(cert['parameters'], dtype=np.float64)
        obj = PairwiseObjective(features, lam)
        with threadpool_limits(limits=1):
            _, g = obj.value_gradient(a)
        norm = np.linalg.norm(g)
        b, error = float(norm * norm / (2 * lam)), float(norm / lam)
        if b > RULE['certificate_B'] or error > RULE['parameter_error_bound']:
            raise ValueError('refined Newton fails recomputed strict certificate')
        for field, actual in [('certificate_B', b), ('parameter_error_bound', error)]:
            if not np.isfinite(cert[field]) or not np.isclose(cert[field], actual, rtol=1e-7, atol=1e-20):
                raise ValueError('reported Newton certificate differs from recomputed certificate')
        if not np.isfinite(cert['refinement_seconds']) or cert['refinement_seconds'] < 0:
            raise ValueError('invalid refinement cost')
        certs[seed, lam] = dict(parameters=a, path=str(cp), training_cache=str(extra_paths[0]), training_metadata=str(extra_paths[1]), run_manifest=str(extra_paths[2]), frozen_lambdas=run['frozen_lambdas'], certificate_B=b, parameter_error_bound=error,
                                refinement_seconds=cert['refinement_seconds'])
    lambdas = sorted({lam for _, lam in certs})
    if len(certs) != 12 or len(lambdas) != 4 or set(certs) != {(s, l) for s in RULE['seeds'] for l in lambdas}:
        raise ValueError('E2 requires all four lambdas times three seeds')
    if any(set(c['frozen_lambdas']) != set(lambdas) for c in certs.values()):
        raise ValueError('candidate lambdas differ from frozen campaign grid')
    if len({c['run_manifest'] for c in certs.values()}) != 1:
        raise ValueError('E2 certificates must come from one campaign')
    # A lambda must refer to the same seed training cache, regardless of lambda.
    for seed in RULE['seeds']:
        caches = {(certs[seed, l]['training_cache'], certs[seed, l]['training_metadata']) for l in lambdas}
        if len(caches) != 1:
            raise ValueError('lambda candidates use different training caches')
    return dict(manifest=m, paths=paths, sources=sources, splits=splits, documents=docs,
                queries=queries, doc_ids=dids, query_ids=qids, train=train, certs=certs, lambdas=lambdas)


def _check_qrels(qrels, doc_ids):
    for labels in qrels.values():
        if not set(labels) <= set(doc_ids) or not any(v > 0 for v in labels.values()):
            raise ValueError('qrels need positive documents present in embedding cache')


def _json_selection(result):
    return {k: v for k, v in result.items() if k not in ('means', 'best_comparisons', 'baseline_comparisons')} | {
        'means': [{'lambda': k, 'value': v} for k, v in result['means'].items()],
        'best_comparisons': [{'lambda': k, **v} for k, v in result['best_comparisons'].items()],
        'baseline_comparisons': [{'lambda': k, **v} for k, v in result['baseline_comparisons'].items()]}


def _run_split(data, split, lambdas, output):
    ids = sorted(data['splits'][split])
    qrels = {q: data['train'][q] for q in ids} if split == 'validation' else read_qrels(data['paths']['test_qrels'])
    if set(qrels) != set(ids):
        raise ValueError('evaluation qrels must exactly match split IDs')
    _check_qrels(qrels, data['doc_ids'])
    indices = bootstrap_indices(ids, split)
    bootstrap_path = output / f'{split}-bootstrap.npz'
    if bootstrap_path.exists():
        with np.load(bootstrap_path, allow_pickle=False) as cache:
            if not np.array_equal(cache['indices'], indices) or cache['query_ids'].tolist() != ids or cache['seed'] != RULE[f'{split}_seed']:
                raise ValueError('saved bootstrap differs from registered sampling')
    else:
        with bootstrap_path.open('xb') as stream:
            np.savez_compressed(stream, indices=indices, query_ids=np.array(ids), seed=RULE[f'{split}_seed'])
    systems = []
    for lam, seed in [(None, None)] + [(l, s) for l in lambdas for s in RULE['seeds']]:
        adapter = None if lam is None else data['certs'][seed, lam]['parameters']
        system_path = output / f'{split}-system-{lam}-{seed}.json'
        receipt_path = system_path.with_suffix('.receipt.json')
        if system_path.exists() or receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt != dict(digest=_digest(system_path), query_ids=ids, regularization=lam, seed=seed):
                raise ValueError('saved system checkpoint is incomplete or changed')
            saved = read_json(system_path)
            if saved['regularization'] != lam or saved['seed'] != seed:
                raise ValueError('saved checkpoint identifies a different system')
            systems.append(saved)
            continue
        retrieval_start = perf_counter()
        ranks = retrieve(data['documents'], data['queries'], data['doc_ids'], data['query_ids'], ids, adapter)
        retrieval_seconds = perf_counter() - retrieval_start
        metrics = per_query_metrics(qrels, ranks)
        aggregate = evaluate(qrels, ranks)
        vectors = {metric: [metrics[q][metric] for q in ids] for metric in METRICS}
        for metric in METRICS:
            if round(float(np.mean(vectors[metric])), 5) != aggregate[metric]:
                raise ValueError('unrounded query metrics disagree with BEIR aggregation')
        record = dict(regularization=lam, seed=seed, metrics=vectors, retrieval_seconds=retrieval_seconds,
                            refinement_seconds=0. if lam is None else data['certs'][seed, lam]['refinement_seconds'],
                            aggregate={metric: float(np.mean(vectors[metric])) for metric in METRICS},
                            beir_rounded={metric: aggregate[metric] for metric in METRICS},
                            rankings={q: list(ranks[q]) for q in ids},
                            parameter_source=None if lam is None else data['certs'][seed, lam]['path'])
        write_json(system_path, record)
        write_json(receipt_path, dict(digest=_digest(system_path), query_ids=ids, regularization=lam, seed=seed))
        systems.append(record)
    baseline = systems[0]
    comparisons = []
    for lam in lambdas:
        group = [s for s in systems if s['regularization'] == lam]
        for metric in METRICS:
            values = np.mean([s['metrics'][metric] for s in group], axis=0)
            comparisons.append(dict(regularization=lam, metric=metric,
                                    **paired_summary(values - baseline['metrics'][metric], indices)))
        for s in group:
            s['ranking_changes_vs_baseline'] = {q: s['rankings'][q] != baseline['rankings'][q] for q in ids}
    result = dict(split=split, query_ids=ids, rule=RULE, systems=systems, comparisons=comparisons)
    if split == 'validation':
        candidates = {None: (ids, baseline['metrics']['nDCG@10'])}
        for lam in lambdas:
            candidates[lam] = (ids, [s['metrics']['nDCG@10'] for s in systems if s['regularization'] == lam])
        result['selection'] = _json_selection(select_systems(candidates, indices, expected_query_ids=ids))
    return result


def validate(manifest, output):
    output = Path(output)
    if output.exists():
        if (output / 'validation.json').exists() or (output / 'freeze.json').exists():
            raise FileExistsError('completed validation is never reselected')
        data = _restore(manifest, output)
    else:
        data = _load(manifest)
        output.mkdir(parents=True)
        write_json(output / 'inputs.json', dict(manifest=str(Path(manifest).resolve()), sources=data['sources'], rule=RULE))
    result = _run_split(data, 'validation', data['lambdas'], output)
    write_json(output / 'validation.json', result)
    return result


def _restore(manifest, output):
    data = _load(manifest)
    saved = read_json(output / 'inputs.json')
    if saved != dict(manifest=str(Path(manifest).resolve()), sources=data['sources'], rule=RULE):
        raise ValueError('frozen evaluation inputs or protocol changed')
    return data


def freeze(manifest, output):
    output = Path(output)
    _restore(manifest, output)
    if (output / 'test.json').exists() or (output / 'test-started.json').exists():
        raise ValueError('cannot freeze or change selection after test began')
    validation = read_json(output / 'validation.json')
    s = validation['selection']
    record = dict(rule=RULE, inputs_digest=_digest(output / 'inputs.json'),
                  validation_digest=_digest(output / 'validation.json'),
                  validation_bootstrap_digest=_digest(output / 'validation-bootstrap.npz'),
                  primary=s['primary'], secondary=s['secondary'], test_lambdas=s['test_lambdas'],
                  test_retrieval_count=s['test_retrieval_count'])
    # Exclusive creation makes freeze immutable to this pipeline.
    write_json(output / 'freeze.json', record)
    return record


def test(manifest, output):
    output = Path(output)
    frozen = read_json(output / 'freeze.json')  # checked before loading any test qrels
    data = _restore(manifest, output)
    expected = dict(inputs_digest=_digest(output / 'inputs.json'), validation_digest=_digest(output / 'validation.json'),
                    validation_bootstrap_digest=_digest(output / 'validation-bootstrap.npz'))
    if frozen['rule'] != RULE or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('freeze no longer matches validation inputs/results')
    s = read_json(output / 'validation.json')['selection']
    if any(frozen[k] != s[k] for k in ('primary', 'secondary', 'test_lambdas', 'test_retrieval_count')):
        raise ValueError('frozen primary/secondary identity was changed')
    if (output / 'test.json').exists():
        raise FileExistsError('completed test cannot be rerun or overwritten')
    start = dict(freeze_digest=_digest(output / 'freeze.json'), test_qrels_digest=_digest(data['paths']['test_qrels']))
    if (output / 'test-started.json').exists():
        if read_json(output / 'test-started.json') != start:
            raise ValueError('test inputs or freeze changed after test began')
    else:
        write_json(output / 'test-started.json', start)
    result = _run_split(data, 'test', frozen['test_lambdas'], output)
    result['frozen_systems'] = frozen
    result['test_qrels_digest'] = start['test_qrels_digest']
    write_json(output / 'test.json', result)
    return result


def summary(manifest, output):
    output = Path(output)
    data = _restore(manifest, output)
    frozen = read_json(output / 'freeze.json')
    result = read_json(output / 'test.json')
    if result['frozen_systems'] != frozen or read_json(output / 'test-started.json')['freeze_digest'] != _digest(output / 'freeze.json'):
        raise ValueError('test/freeze provenance differs')
    if result['test_qrels_digest'] != _digest(data['paths']['test_qrels']):
        raise ValueError('test qrels changed since testing')
    record = dict(primary=frozen['primary'], secondary=frozen['secondary'], merged=frozen['primary'] == frozen['secondary'],
                  retrieval_count=len(result['systems']), query_count=len(result['query_ids']),
                  per_seed=[{k: s[k] for k in ('regularization', 'seed', 'aggregate')} for s in result['systems']],
                  comparisons=result['comparisons'],
                  retrieval_seconds=sum(s['retrieval_seconds'] for s in result['systems']),
                  refinement_seconds=sum(s['refinement_seconds'] for s in result['systems']))
    write_json(output / 'summary.json', record)
    return record


def build_manifest(campaign, manifest, embeddings, splits, train_qrels, test_qrels, *, synthetic=False):
    """Build E2 input from campaign's completed E1, without inspecting test labels."""
    campaign = Path(campaign).resolve()
    run = read_json(campaign / 'manifest.json')
    completed = read_json(campaign / 'e1-summary.json')
    mode = 'synthetic' if synthetic else 'formal'
    if run['mode'] != mode or completed['mode'] != mode or completed['stage'] != 'e1' or completed['state'] not in ('complete', 'complete_with_failures'):
        raise ValueError('E2 manifest requires matching completed E1 campaign')
    targets = completed['targets']
    certs = [str(_path(campaign, t['certified_newton'])) for t in targets]
    if len(certs) != 12 or len(set(certs)) != 12 or not all(Path(p).is_file() for p in certs):
        raise ValueError('completed campaign must provide twelve unique refined Newton files')
    record = dict(embeddings=str(Path(embeddings).resolve()), splits=str(Path(splits).resolve()),
                  train_qrels=str(Path(train_qrels).resolve()), test_qrels=str(Path(test_qrels).resolve()),
                  certificates=certs, synthetic=synthetic)
    write_json(manifest, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('manifest', 'validate', 'freeze', 'test', 'summary'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    for key in ('campaign', 'embeddings', 'splits', 'train-qrels', 'test-qrels'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.command == 'manifest':
        required = ('campaign', 'embeddings', 'splits', 'train_qrels', 'test_qrels')
        if any(getattr(args, k) is None for k in required):
            parser.error('manifest needs --campaign --embeddings --splits --train-qrels --test-qrels')
        result = build_manifest(args.campaign, args.manifest, args.embeddings, args.splits, args.train_qrels, args.test_qrels, synthetic=args.synthetic)
    else:
        if args.output is None:
            parser.error('--output is required')
        result = globals()[args.command](args.manifest, args.output)
    print(json.dumps({'command': args.command, 'output': str(args.output), 'complete': bool(result)}))


if __name__ == '__main__':
    main()
