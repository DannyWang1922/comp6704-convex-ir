"""Independent dispatch audit: synthetic arrays only, no solve or retrieval."""
from pathlib import Path
import numpy as np
from query_adapt import campaign as c
from query_adapt.data import read_json, write_json


def test_e3_dispatch_preserves_e1_full_target_and_nested_row_identity(tmp_path, monkeypatch):
    full = np.arange(5936 * 2, dtype=float).reshape(5936, 2)
    rows = [{'query_id': str(i), 'kind': 'hard'} for i in range(5936)]
    cache, metadata = tmp_path / 'cache.npz', tmp_path / 'rows.json'
    np.savez(cache, features=full)
    write_json(metadata, {'triples': rows})
    cfg = {'e1_seconds': 30., 'e3_seconds': 30., 'reference_seconds': 2., 'solver': {}}
    provenance = {'synthetic': True, 'caches': {'42': {'cache': str(cache), 'metadata': str(metadata)}}}
    run = c.Campaign(cfg, tmp_path / 'campaign', provenance, 'synthetic')
    completed = [dict(seed=s, regularization=l, n=5936) for s in (42, 43, 44) for l in (.01, .001, .0001, .00001)]
    write_json(run.output / 'e1-summary.json', {'stage': 'e1', 'mode': 'formal', 'state': 'complete', 'targets': completed})
    calls, benchmarks = [], []

    def target(seed, lam, x, selected_rows, cache_path, meta_path, stage, rate):
        saved = read_json(meta_path)
        ix = saved['selected_indices']
        np.testing.assert_array_equal(x, full[ix])
        assert selected_rows == [rows[i] for i in ix]
        assert read_json(meta_path)['subset_seed'] == 2026
        calls.append(ix)
        directory = run.output / f'audit-{len(x)}'
        directory.mkdir()
        write_json(directory / 'diagnostics.json', {'seconds': 0.})
        return dict(seed=seed, regularization=lam, n=len(x), state='complete', timing={}, certified_newton=str(directory / 'certified-newton.json'))

    monkeypatch.setattr(run, 'target', target)
    monkeypatch.setattr(run, 'microbench', lambda directory, obj, stage: benchmarks.append((Path(directory).name, obj.features.copy())))
    monkeypatch.setattr(c, 'generate_figures', lambda *args: None)
    run.run('e3')
    assert list(map(len, calls)) == [512, 2048]
    assert calls[0] == calls[1][:512]
    assert len(benchmarks) == 1
    assert benchmarks[0][0] == c.target_id(42, .001, 5936)
    np.testing.assert_array_equal(benchmarks[0][1], full)
