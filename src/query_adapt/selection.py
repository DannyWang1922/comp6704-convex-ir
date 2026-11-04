"""Pure query-level metrics and preregistered paired selection utilities.

No retrieval or file access is performed. None denotes the identity baseline.
"""
import numpy as np

VALIDATION_SEED = 20261101
TEST_SEED = 20261102
N_BOOTSTRAP = 2000


def align_queries(query_ids, values, expected_ids=None):
    """Return lexically sorted IDs and aligned finite values; reject duplicates."""
    ids = list(query_ids)
    if not ids or any(not isinstance(q, str) for q in ids) or len(set(ids)) != len(ids):
        raise ValueError("query IDs must be nonempty unique strings")
    if expected_ids is not None:
        expected = list(expected_ids)
        if len(set(expected)) != len(expected) or set(ids) != set(expected):
            raise ValueError("query IDs must match exactly")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != len(ids) or not np.isfinite(array).all():
        raise ValueError("finite values must have query dimension last")
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return tuple(ids[i] for i in order), array[..., order]


def bootstrap_indices(query_ids, split="validation", n_resamples=N_BOOTSTRAP):
    """One index matrix per split, reused for every candidate and metric."""
    ids, _ = align_queries(query_ids, np.zeros(len(query_ids)))
    if tuple(query_ids) != ids:
        raise ValueError("bootstrap query IDs must already be sorted")
    if split not in ("validation", "test") or n_resamples < 2:
        raise ValueError("invalid split or resample count")
    seed = VALIDATION_SEED if split == "validation" else TEST_SEED
    return np.random.default_rng(seed).integers(0, len(ids), size=(n_resamples, len(ids)))


def paired_summary(differences, indices):
    """Mean, bootstrap SE (ddof=1), and percentile 95% interval."""
    d = np.asarray(differences, dtype=np.float64)
    ix = np.asarray(indices)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError("differences must be a finite nonempty vector")
    if ix.ndim != 2 or ix.shape[0] < 2 or ix.shape[1] != len(d) or not np.issubdtype(ix.dtype, np.integer) or np.any(ix < 0) or np.any(ix >= len(d)):
        raise ValueError("invalid shared bootstrap indices")
    means = d[ix].mean(axis=1)
    return {"mean": float(d.mean()), "se": float(means.std(ddof=1)),
            "ci95": tuple(float(x) for x in np.percentile(means, [2.5, 97.5], method="linear"))}


def select_systems(candidates, indices, *, expected_query_ids=None):
    """Select from baseline plus four lambdas using seed-averaged nDCG@10.

    Mapping values are (query_ids, values), with baseline a query vector and
    finite lambdas exactly three seed rows. Callers supply unrounded nDCG only.
    indices must come from bootstrap_indices(sorted_query_ids) for this exact
    query set, reused across metrics. Pass those same sorted IDs through
    expected_query_ids to check the association; a bare matrix cannot encode IDs.
    """
    if None not in candidates or len(candidates) != 5:
        raise ValueError("expected baseline and four finite lambdas")
    finite = [k for k in candidates if k is not None]
    if any(not isinstance(k, (int, float)) or not np.isfinite(k) or k <= 0 for k in finite):
        raise ValueError("lambdas must be positive finite numbers")
    ids = None
    vectors = {}
    for key, (qids, values) in candidates.items():
        aligned_ids, a = align_queries(qids, values, ids)
        ids = aligned_ids
        if (key is None and a.ndim != 1) or (key is not None and a.shape != (3, len(ids))):
            raise ValueError("baseline needs one vector; lambda needs three seed rows")
        if np.any(a < 0) or np.any(a > 1):
            raise ValueError("nDCG must lie in [0, 1]")
        vectors[key] = a if key is None else a.mean(axis=0)
    if expected_query_ids is not None and tuple(expected_query_ids) != ids:
        raise ValueError("bootstrap query IDs must match sorted candidate IDs")
    means = {key: float(v.mean()) for key, v in vectors.items()}
    preference = lambda key: (key is None, float('inf') if key is None else key)
    best = max(means, key=lambda key: (means[key], preference(key)))
    comparisons = {key: paired_summary(vectors[best] - v, indices) for key, v in vectors.items()}
    eligible = [key for key, s in comparisons.items() if s['mean'] <= s['se']]
    primary = max(eligible, key=preference)
    secondary = max(finite, key=lambda key: (means[key], key))
    adapters = sorted({k for k in (primary, secondary) if k is not None}, reverse=True)
    return {"query_ids": ids, "means": means, "best": best, "eligible": eligible,
            "primary": primary, "secondary": secondary, "test_lambdas": adapters,
            "test_retrieval_count": 1 + 3 * len(adapters), "best_comparisons": comparisons,
            "baseline_comparisons": {key: paired_summary(v - vectors[None], indices)
                                     for key, v in vectors.items() if key is not None}}


def per_query_metrics(qrels, results):
    """Unrounded pytrec_eval values, preserving query/document namespaces."""
    import pytrec_eval
    if not qrels or set(qrels) != set(results):
        raise ValueError("evaluation query IDs must match exactly")
    raw = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.100'}).evaluate(results)
    if set(raw) != set(qrels):
        raise ValueError("evaluator omitted queries")
    return {qid: {"nDCG@10": raw[qid]['ndcg_cut_10'],
                  "Recall@100": raw[qid]['recall_100']} for qid in sorted(raw)}
