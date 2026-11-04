import numpy as np
import pytest
from query_adapt.data import split_queries, make_pairs
from query_adapt.retrieval import retrieve, evaluate


def test_query_split_is_disjoint_and_reproducible():
    train = {str(i): {"doc": 1} for i in range(20)}
    test = {"heldout": {"doc": 1}}
    split = split_queries(train, test, 0.2, 42)
    assert split == split_queries(train, test, 0.2, 42)
    assert len(split["validation"]) == 4
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == set(train)
    assert not (set(split["train"]) | set(split["validation"])) & set(split["test"])
    with pytest.raises(ValueError):
        split_queries(train, {"1": {"doc": 1}}, 0.2, 42)


def test_pairs_exclude_all_positives_and_match_score_difference():
    rng = np.random.default_rng(8)
    docs, queries = rng.normal(size=(10, 4)), rng.normal(size=(2, 4))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    dids, qids = [f"d{i}" for i in range(10)], ["train", "validation"]
    qrels = {"train": {"d0": 1, "d1": 1}, "validation": {"d2": 1}}
    args = (docs, queries, dids, qids, qrels, ["train"])
    features, rows = make_pairs(*args, random_negatives=2, hard_negatives=2)
    repeated, repeated_rows = make_pairs(*args, random_negatives=2, hard_negatives=2)
    np.testing.assert_array_equal(features, repeated)
    assert rows == repeated_rows and len(rows) == 8
    a = rng.normal(size=4)
    for x, row in zip(features, rows):
        assert row["query_id"] == "train"
        assert row["negative_id"] not in {"d0", "d1"}
        positive, negative = dids.index(row["positive_id"]), dids.index(row["negative_id"])
        assert a @ x == pytest.approx(((a * queries[0]) @ docs[positive]
                                      - (a * queries[0]) @ docs[negative]) / 0.1)
    for positive in ("d0", "d1"):
        negatives = [r["negative_id"] for r in rows if r["positive_id"] == positive]
        assert len(negatives) == len(set(negatives))


def test_identity_ranking_and_hand_computed_metrics():
    docs = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    queries = np.array([[1., 0.]])
    result = retrieve(docs, queries, ["same", "second", "third"], ["same"], ["same"])
    adapted = retrieve(docs, queries, ["same", "second", "third"], ["same"], ["same"], np.ones(2))
    assert result == adapted
    assert list(result["same"]) == ["same", "second", "third"]
    # Equal query/document IDs must not silently remove the relevant document.
    metric = evaluate({"same": {"same": 1}}, result)
    assert metric["nDCG@10"] == 1
    assert metric["Recall@100"] == 1
    second_only = evaluate({"same": {"second": 1}}, result)
    assert second_only["nDCG@10"] == pytest.approx(1 / np.log2(3), abs=1e-5)
