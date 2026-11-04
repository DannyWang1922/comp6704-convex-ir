"""Exact full-corpus dot-product retrieval and BEIR evaluation."""
import numpy as np


def retrieve(documents, queries, doc_ids, query_ids, selected_ids, adapter=None, top_k=100):
    a = np.ones(documents.shape[1]) if adapter is None else np.asarray(adapter)
    if a.shape != (documents.shape[1],) or not np.isfinite(a).all():
        raise ValueError("invalid adapter")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    qi = {qid: index for index, qid in enumerate(query_ids)}
    results = {}
    for qid in selected_ids:
        scores = documents.astype(np.float64, copy=False) @ (queries[qi[qid]] * a)
        indices = np.argsort(-scores, kind="stable")[:min(top_k, len(doc_ids))]
        results[qid] = {doc_ids[i]: float(scores[i]) for i in indices}
    return results


def evaluate(qrels, results):
    from beir.retrieval.evaluation import EvaluateRetrieval
    if set(qrels) != set(results):
        raise ValueError("evaluation query IDs must match exactly")
    # Query and document IDs are separate namespaces in SciFact.
    ndcg, mean_ap, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [10, 100], ignore_identical_ids=False)
    return {"nDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"],
            "details": {"ndcg": ndcg, "map": mean_ap, "recall": recall, "precision": precision}}
