"""Explicit preparation stages. This module never optimizes an adapter."""
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import numpy as np
from .data import (read_json, write_json, load_data, split_queries,
                   validate_embeddings, make_pairs)
from .retrieval import retrieve, evaluate


def paths(config, root):
    artifacts = root / config["paths"]["artifacts"]
    artifacts.mkdir(parents=True, exist_ok=True)
    # Keep public caches local and avoid ambient Hub credentials.
    os.environ.setdefault("HF_HOME", str(artifacts / "hf-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return artifacts


def download(config, root):
    artifacts = paths(config, root)
    source = config["data"]["url"]
    archive = artifacts / "downloads/scifact.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        print("Downloading SciFact", flush=True)
        with urllib.request.urlopen(source, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        if not zipfile.is_zipfile(partial):
            raise ValueError("download is not a ZIP archive")
        partial.rename(archive)
    data = artifacts / "data"
    if not (data / "scifact").exists():
        data.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (data / name).resolve()
                if not target.is_relative_to(data.resolve()):
                    raise ValueError("unsafe ZIP path")
            z.extractall(data)
    corpus, queries, train, test = load_data(data / "scifact")
    info = artifacts / "data/source.json"
    if not info.exists():
        write_json(info, {"url": source, "documents": len(corpus), "query_texts": len(queries),
                          "train_queries": len(train), "test_queries": len(test)})
    split_path = artifacts / "data/splits.json"
    splits = split_queries(train, test, config["data"]["validation_fraction"], config["data"]["split_seed"])
    if split_path.exists():
        if read_json(split_path) != splits:
            raise ValueError("existing splits differ; use a new artifacts directory")
    else:
        write_json(split_path, splits)
    from huggingface_hub import HfApi, snapshot_download
    manifest_path = artifacts / "model-source.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest["model_id"] != config["model"]["id"]
                or config["model"]["revision"] not in {manifest["requested_revision"], manifest["revision"]}):
            raise ValueError("existing model differs; use a new artifacts directory")
        revision = manifest["revision"]
    else:
        revision = HfApi().model_info(config["model"]["id"], revision=config["model"]["revision"], token=False).sha
    print(f"Downloading model revision {revision}", flush=True)
    snapshot_download(repo_id=config["model"]["id"], revision=revision,
                      local_dir=artifacts / "model", token=False,
                      allow_patterns=["*.json", "*.safetensors", "vocab.txt", "README.md"],
                      ignore_patterns=["onnx/*", "openvino/*"], max_workers=2)
    if not manifest_path.exists():
        write_json(manifest_path, {"model_id": config["model"]["id"],
                                  "requested_revision": config["model"]["revision"], "revision": revision})
    return {"documents": len(corpus), "train": len(splits["train"]),
            "validation": len(splits["validation"]), "test": len(splits["test"]), "model_revision": revision}


def encode(config, root):
    artifacts = paths(config, root)
    output = artifacts / "embeddings.npz"
    metadata = artifacts / "embeddings.json"
    if output.exists() or metadata.exists():
        raise FileExistsError("embeddings already exist; use a new artifacts directory")
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(config["model"]["threads"])
    corpus, queries, _, _ = load_data(artifacts / "data/scifact")
    source = read_json(artifacts / "model-source.json")
    model = SentenceTransformer(str(artifacts / "model"), device=config["model"]["device"],
                                local_files_only=True, trust_remote_code=False)
    model.max_seq_length = config["model"]["max_seq_length"]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dids, qids = sorted(corpus), sorted(queries)
    texts = [(corpus[i].get("title", "") + " " + corpus[i]["text"]).strip() for i in dids]
    start = time.perf_counter()
    kwargs = dict(batch_size=config["model"]["batch_size"], normalize_embeddings=True,
                  convert_to_numpy=True, show_progress_bar=True)
    documents = model.encode(texts, **kwargs)
    vectors = model.encode([queries[i]["text"] for i in qids], **kwargs)
    elapsed = time.perf_counter() - start
    validate_embeddings(documents, vectors)
    if documents.shape[1] != 384:
        raise ValueError("expected 384-dimensional embeddings")
    with output.open("xb") as stream:
        np.savez_compressed(stream, documents=documents, queries=vectors,
                            doc_ids=np.array(dids), query_ids=np.array(qids))
    result = {"source": source, "model_config": config["model"], "documents": len(dids),
              "queries": len(qids), "dimension": 384, "encoding_seconds": elapsed,
              "text_format": "(title + ' ' + text).strip()", "dtype": str(documents.dtype)}
    write_json(metadata, result)
    return result


def load_embeddings(config, root):
    artifacts = paths(config, root)
    info = read_json(artifacts / "embeddings.json")
    if info["model_config"] != config["model"]:
        raise ValueError("model configuration differs from cached embeddings")
    if info["source"] != read_json(artifacts / "model-source.json"):
        raise ValueError("model source differs from cached embeddings")
    with np.load(artifacts / "embeddings.npz", allow_pickle=False) as cache:
        docs, queries = cache["documents"], cache["queries"]
        dids, qids = cache["doc_ids"].tolist(), cache["query_ids"].tolist()
    validate_embeddings(docs, queries)
    if len(dids) != len(docs) or len(qids) != len(queries):
        raise ValueError("embedding IDs do not align")
    corpus, raw_queries, train, test = load_data(artifacts / "data/scifact")
    if dids != sorted(corpus) or qids != sorted(raw_queries):
        raise ValueError("embedding IDs differ from source")
    splits = read_json(artifacts / "data/splits.json")
    expected = split_queries(train, test, config["data"]["validation_fraction"], config["data"]["split_seed"])
    if splits != expected:
        raise ValueError("split configuration differs from prepared data")
    return docs, queries, dids, qids, splits, train


def baseline(config, root):
    artifacts = paths(config, root)
    output = artifacts / "baseline-validation.json"
    if output.exists():
        raise FileExistsError(output)
    docs, queries, dids, qids, splits, train = load_embeddings(config, root)
    results = retrieve(docs, queries, dids, qids, splits["validation"])
    metrics = evaluate({q: train[q] for q in splits["validation"]}, results)
    write_json(output, {"split": "validation", "query_count": len(results), "metrics": metrics,
                        "adapter": "identity", "ranking": results})
    return metrics


def pairs(config, root):
    artifacts = paths(config, root)
    output = artifacts / "train-pairs.npz"
    if output.exists() or (artifacts / "train-pairs.json").exists():
        raise FileExistsError("pairs already exist; use a new artifacts directory")
    docs, queries, dids, qids, splits, train = load_embeddings(config, root)
    features, rows = make_pairs(docs, queries, dids, qids, train, splits["train"], **config["pairs"])
    with output.open("xb") as stream:
        np.savez_compressed(stream, features=features)
    write_json(artifacts / "train-pairs.json", {"config": config["pairs"], "split": "train",
                                                "count": len(rows), "triples": rows})
    return {"training_pairs": len(rows), "dimension": features.shape[1], "dtype": str(features.dtype)}


def verify(config, root):
    artifacts = paths(config, root)
    docs, queries, dids, qids, splits, train = load_embeddings(config, root)
    saved = read_json(artifacts / "train-pairs.json")
    if saved["config"] != config["pairs"]:
        raise ValueError("pair configuration differs")
    with np.load(artifacts / "train-pairs.npz", allow_pickle=False) as cache:
        features = cache["features"]
    expected, rows = make_pairs(docs, queries, dids, qids, train, splits["train"], **config["pairs"])
    if rows != saved["triples"] or not np.array_equal(features, expected):
        raise ValueError("training features or triple IDs do not reproduce")
    baseline_result = read_json(artifacts / "baseline-validation.json")
    ranking = retrieve(docs, queries, dids, qids, splits["validation"])
    metrics = evaluate({q: train[q] for q in splits["validation"]}, ranking)
    if baseline_result["ranking"] != ranking or baseline_result["metrics"] != metrics:
        raise ValueError("baseline does not reproduce")
    return {"status": "verified", "documents": len(dids), "queries": len(qids),
            "training_queries": len(splits["train"]), "validation_queries": len(splits["validation"]),
            "test_queries_reserved": len(splits["test"]), "training_pairs": len(rows),
            "baseline_validation": metrics, "adapter_training_run": False, "test_evaluation_run": False}
