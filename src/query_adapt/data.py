"""SciFact input validation, reproducible query splits and fixed pair features."""
import csv
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse silent overwrites of inputs, splits and results.
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(serialized)


def read_jsonl(path):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row["_id"])
            if key in rows:
                raise ValueError(f"duplicate ID: {key}")
            rows[key] = row
    return rows


def read_qrels(path):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
            if did in result.get(qid, {}):
                raise ValueError(f"duplicate qrel: {qid}/{did}")
            result.setdefault(qid, {})[did] = score
    return result


def load_data(directory):
    directory = Path(directory)
    corpus = read_jsonl(directory / "corpus.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    train = read_qrels(directory / "qrels/train.tsv")
    test = read_qrels(directory / "qrels/test.tsv")
    if set(train) & set(test):
        raise ValueError("official train/test queries overlap")
    for labels in (train, test):
        for qid, docs in labels.items():
            if qid not in queries or not set(docs) <= corpus.keys():
                raise ValueError("qrels refer to missing query or document")
            if not any(score > 0 for score in docs.values()):
                raise ValueError("query has no positive document")
    return corpus, queries, train, test


def split_queries(train_qrels, test_qrels, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    ids = np.array(sorted(train_qrels))
    if len(ids) < 2 or set(train_qrels) & set(test_qrels):
        raise ValueError("invalid official query partitions")
    ids = np.random.default_rng(seed).permutation(ids)
    size = max(1, min(len(ids) - 1, round(len(ids) * validation_fraction)))
    return {"train": sorted(ids[size:].tolist()), "validation": sorted(ids[:size].tolist()),
            "test": sorted(test_qrels), "seed": seed, "validation_fraction": validation_fraction}


def validate_embeddings(documents, queries):
    for x in (documents, queries):
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("invalid embedding matrix")
        if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
            raise ValueError("embeddings must be normalized")
    if documents.shape[1] != queries.shape[1]:
        raise ValueError("embedding dimensions differ")


def make_pairs(documents, queries, doc_ids, query_ids, qrels, train_ids,
               random_negatives=4, hard_negatives=4, seed=42, temperature=0.1):
    """Exclude every known positive; sample disjoint hard and random negatives."""
    validate_embeddings(documents, queries)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive")
    if min(random_negatives, hard_negatives) < 0 or random_negatives + hard_negatives == 0:
        raise ValueError("invalid negative counts")
    if len(set(doc_ids)) != len(doc_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("duplicate vector IDs")
    di = {key: i for i, key in enumerate(doc_ids)}
    qi = {key: i for i, key in enumerate(query_ids)}
    rng = np.random.default_rng(seed)
    rows, features = [], []
    for qid in sorted(train_ids):
        positives = sorted(d for d, relevance in qrels[qid].items() if relevance > 0)
        positive_indices = {di[d] for d in positives}
        scores = documents @ queries[qi[qid]]
        # Stable sort makes score ties deterministic for fixed doc_ids.
        eligible = [int(i) for i in np.argsort(-scores, kind="stable") if i not in positive_indices]
        if len(eligible) < random_negatives + hard_negatives:
            raise ValueError("not enough distinct negatives")
        hard = eligible[:hard_negatives]
        pool = eligible[hard_negatives:]
        for positive in positives:
            random = rng.choice(pool, size=random_negatives, replace=False).tolist()
            for negative, kind in [(i, "hard") for i in hard] + [(i, "random") for i in random]:
                rows.append({"query_id": qid, "positive_id": positive,
                             "negative_id": doc_ids[negative], "kind": kind})
                features.append(queries[qi[qid]].astype(np.float64)
                                * (documents[di[positive]].astype(np.float64)
                                   - documents[negative].astype(np.float64)) / temperature)
    if not rows:
        raise ValueError("no training pairs")
    return np.stack(features), rows
